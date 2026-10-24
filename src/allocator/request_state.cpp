#include "kvlab/request_state.hpp"

#include <exception>
#include <set>
#include <stdexcept>
#include <utility>

namespace kvlab {
namespace {

std::pmr::memory_resource* checked_resource(std::pmr::memory_resource* resource) {
  if (resource == nullptr) {
    throw std::invalid_argument("RequestState needs a non-null memory resource");
  }
  return resource;
}

std::size_t page_count_for(std::size_t tokens, std::size_t page_tokens) noexcept {
  // Avoid tokens + page_tokens - 1 overflowing. page_tokens is positive.
  return tokens / page_tokens + (tokens % page_tokens != 0);
}

}  // namespace

RequestState::RequestState(PagePool& pool, std::pmr::memory_resource* resource)
    : pool_(&pool), pages_(checked_resource(resource)) {}

RequestState::RequestState(RequestState&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      pages_(std::move(other.pages_)),
      length_(std::exchange(other.length_, 0)),
      reserved_length_(std::exchange(other.reserved_length_, 0)) {
  other.pages_.clear();
}

RequestState& RequestState::operator=(RequestState&& other) {
  if (this != &other) {
    // PMR vector allocators do not propagate on assignment. Prepare storage using
    // our resource; all potentially throwing work precedes either side's mutation.
    Table candidate{pages_.get_allocator().resource()};
    candidate.resize(other.pages_.size());
    for (std::size_t i = 0; i < candidate.size(); ++i) {
      candidate[i] = std::move(other.pages_[i]);
    }
    pages_.swap(candidate);  // Same resource; old destination leases retire at scope exit.
    pool_ = std::exchange(other.pool_, nullptr);
    length_ = std::exchange(other.length_, 0);
    reserved_length_ = std::exchange(other.reserved_length_, 0);
    other.pages_.clear();
  }
  return *this;
}

RequestStatus RequestState::reserve_for_length(std::size_t target) {
  if (pool_ == nullptr) {
    return RequestStatus::MovedFrom;
  }
  if (target <= reserved_length_) {
    return RequestStatus::Ok;
  }
  const auto needed = page_count_for(target, pool_->tokens_per_page());
  const auto additional = needed - pages_.size();  // Reservation only grows.
  if (additional > pool_->stats().free_pages) {
    return RequestStatus::PoolExhausted;
  }
  if (additional == 0) {
    reserved_length_ = target;
    return RequestStatus::Ok;
  }
  Table candidate{pages_.get_allocator().resource()};
  candidate.resize(needed);  // Host allocation first; existing table stays untouched.
  for (std::size_t i = pages_.size(); i < needed; ++i) {
    auto acquired = PageLease::try_allocate(*pool_);
    if (!acquired) {
      return RequestStatus::PoolExhausted;
    }
    candidate[i] = std::move(*acquired);
  }
  // No throwing work remains. Existing owners move only at this commit point.
  for (std::size_t i = 0; i < pages_.size(); ++i) {
    candidate[i] = std::move(pages_[i]);
  }
  pages_.swap(candidate);
  reserved_length_ = target;
  return RequestStatus::Ok;
}

RequestStatus RequestState::commit_length(std::size_t target) noexcept {
  if (pool_ == nullptr) {
    return RequestStatus::MovedFrom;
  }
  if (target < length_ || target > reserved_length_) {
    return RequestStatus::InvalidLength;
  }
  if (target == length_) {
    return RequestStatus::Ok;
  }
  const auto page_tokens = pool_->tokens_per_page();
  const auto first = length_ / page_tokens;
  // Check all pages touched by newly committed tokens before sealing any of them.
  for (std::size_t i = first; i < page_count_for(target, page_tokens); ++i) {
    const auto handle = pages_[i].handle();
    const auto info = handle ? pool_->inspect(*handle) : std::nullopt;
    if (!info || !info->writable()) {
      return RequestStatus::NotWritable;
    }
  }
  for (std::size_t i = first; i < target / page_tokens; ++i) {
    // Preflight + single-thread contract guarantee this transition cannot fail.
    if (pool_->seal_full(*pages_[i].handle(), page_tokens) != SealResult::Sealed) {
      std::terminate();
    }
  }
  length_ = target;
  return RequestStatus::Ok;
}

RequestStatus RequestState::share_prefix_from(const RequestState& source,
                                             std::size_t full_pages) {
  if (pool_ == nullptr || source.pool_ == nullptr) {
    return RequestStatus::MovedFrom;
  }
  if (!pages_.empty()) {
    return RequestStatus::NotEmpty;
  }
  if (pool_ != source.pool_) {
    return RequestStatus::PoolMismatch;
  }
  const auto page_tokens = pool_->tokens_per_page();
  if (full_pages > source.length_ / page_tokens) {
    return RequestStatus::InvalidLength;
  }
  Table candidate{pages_.get_allocator().resource()};
  candidate.resize(full_pages);
  for (std::size_t i = 0; i < full_pages; ++i) {
    auto shared = source.pages_[i].try_share();
    if (!shared) {
      return RequestStatus::NotShareable;
    }
    candidate[i] = std::move(*shared);
  }
  pages_.swap(candidate);
  // Bound above proves the multiplication cannot exceed source.length_.
  length_ = full_pages * page_tokens;
  reserved_length_ = length_;
  return RequestStatus::Ok;
}

void RequestState::reset() noexcept {
  pages_.clear();
  length_ = 0;
  reserved_length_ = 0;
}

std::optional<PageHandle> RequestState::page_handle(std::size_t logical_page) const noexcept {
  return logical_page < pages_.size() ? pages_[logical_page].handle() : std::nullopt;
}

std::optional<TokenLocation> RequestState::token_location(std::size_t position) const noexcept {
  if (pool_ == nullptr || position >= length_) {
    return std::nullopt;
  }
  const auto page_tokens = pool_->tokens_per_page();
  return TokenLocation{*pages_[position / page_tokens].handle(), position % page_tokens};
}

void RequestState::validate_invariants() const {
  if (pool_ == nullptr) {
    if (!pages_.empty() || length_ != 0 || reserved_length_ != 0) {
      throw std::logic_error("Moved-from request must be empty");
    }
    return;
  }
  const auto page_tokens = pool_->tokens_per_page();
  if (length_ > reserved_length_ || pages_.size() != page_count_for(reserved_length_, page_tokens)) {
    throw std::logic_error("Request length/table size invariant failed");
  }
  std::set<PageId> seen;
  for (std::size_t i = 0; i < pages_.size(); ++i) {
    const auto handle = pages_[i].handle();
    const auto info = handle ? pool_->inspect(*handle) : std::nullopt;
    if (!info || !seen.insert(handle->id).second) {
      throw std::logic_error("Request contains an invalid or duplicate page");
    }
    if (i < length_ / page_tokens ? !info->sealed : !info->writable()) {
      throw std::logic_error("Request sealed-prefix/writable-tail invariant failed");
    }
  }
}

}  // namespace kvlab
