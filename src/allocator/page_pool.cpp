#include "kvlab/page_pool.hpp"

#include <stdexcept>

namespace kvlab {

PagePool::PagePool(std::size_t capacity, std::size_t tokens_per_page)
    : tokens_per_page_(tokens_per_page) {
  if (tokens_per_page == 0) {
    throw std::invalid_argument("PagePool tokens_per_page must be positive");
  }
  if (capacity > max_capacity()) {
    throw std::length_error("PagePool capacity exceeds the signed page-table limit");
  }
  slots_.resize(capacity);
  free_ids_.reserve(capacity);
  // This initial order makes the demo readable; callers must not depend on allocation order.
  for (std::size_t remaining = capacity; remaining > 0; --remaining) {
    free_ids_.push_back(static_cast<PageId>(remaining - 1));
  }
}

std::optional<PageHandle> PagePool::try_allocate() {
  if (free_ids_.empty()) {
    return std::nullopt;
  }
  const PageId id = free_ids_.back();
  auto& slot = slots_[id];
  if (slot.generation == std::numeric_limits<std::uint64_t>::max()) {
    // Never wrap a generation and accidentally make an old handle valid again.
    throw std::overflow_error("PagePool generation exhausted");
  }
  free_ids_.pop_back();
  ++slot.generation;
  slot.references = 1;
  slot.sealed = false;
  return PageHandle{id, slot.generation};
}

SealResult PagePool::seal_full(PageHandle handle, std::size_t valid_tokens) noexcept {
  if (handle.id >= slots_.size()) {
    return SealResult::InvalidPage;
  }
  auto& slot = slots_[handle.id];
  if (slot.references == 0) {
    return SealResult::NotAllocated;
  }
  if (slot.generation != handle.generation) {
    return SealResult::StaleHandle;
  }
  if (slot.sealed) {
    return SealResult::AlreadySealed;
  }
  if (valid_tokens != tokens_per_page_) {
    return SealResult::NotFull;
  }
  slot.sealed = true;
  return SealResult::Sealed;
}

RetainResult PagePool::retain(PageHandle handle) noexcept {
  if (handle.id >= slots_.size()) {
    return RetainResult::InvalidPage;
  }
  auto& slot = slots_[handle.id];
  if (slot.references == 0) {
    return RetainResult::NotAllocated;
  }
  if (slot.generation != handle.generation) {
    return RetainResult::StaleHandle;
  }
  if (!slot.sealed) {
    return RetainResult::NotSealed;
  }
  if (slot.references == std::numeric_limits<std::uint32_t>::max()) {
    return RetainResult::Overflow;
  }
  ++slot.references;
  return RetainResult::Retained;
}

ReleaseResult PagePool::release(PageHandle handle) noexcept {
  if (handle.id >= slots_.size()) {
    return ReleaseResult::InvalidPage;
  }
  auto& slot = slots_[handle.id];
  if (slot.references == 0) {
    return ReleaseResult::NotAllocated;
  }
  if (slot.generation != handle.generation) {
    return ReleaseResult::StaleHandle;
  }
  if (slot.references > 1) {
    --slot.references;
    return ReleaseResult::Released;
  }
  // Construction reserved capacity entries. A live page has no entry in this stack,
  // so returning it fits without allocation; PageId copying cannot throw.
  free_ids_.push_back(handle.id);
  slot.references = 0;
  slot.sealed = false;
  return ReleaseResult::Released;
}

std::optional<PageInfo> PagePool::inspect(PageHandle handle) const noexcept {
  if (handle.id >= slots_.size()) {
    return std::nullopt;
  }
  const auto& slot = slots_[handle.id];
  if (slot.references == 0 || slot.generation != handle.generation) {
    return std::nullopt;
  }
  return PageInfo{slot.references, slot.sealed};
}

PoolStats PagePool::stats() const noexcept {
  return {slots_.size(), free_ids_.size(), slots_.size() - free_ids_.size()};
}

void PagePool::validate_invariants() const {
  if (free_ids_.size() > slots_.size() || free_ids_.capacity() < slots_.size()) {
    throw std::logic_error("PagePool free-list size/capacity invariant failed");
  }
  std::vector<std::uint8_t> seen(slots_.size(), 0);
  for (const PageId id : free_ids_) {
    if (id >= slots_.size() || seen[id] != 0 || slots_[id].references != 0) {
      throw std::logic_error("PagePool free list contains an invalid, duplicate, or live ID");
    }
    seen[id] = 1;
  }
  for (std::size_t id = 0; id < slots_.size(); ++id) {
    const auto& slot = slots_[id];
    if (slot.references != 0) {
      if (seen[id] != 0 || slots_[id].generation == 0) {
        throw std::logic_error("PagePool live-page invariant failed");
      }
      if (slot.references > 1 && !slot.sealed) {
        throw std::logic_error("PagePool shared page must be sealed");
      }
    } else if (seen[id] != 1 || slot.sealed) {
      throw std::logic_error("PagePool free page is missing or still sealed");
    }
  }
}

}  // namespace kvlab
