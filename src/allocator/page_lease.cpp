#include "kvlab/page_lease.hpp"

#include <exception>
#include <utility>

namespace kvlab {

PageLease::~PageLease() noexcept { reset(); }

PageLease::PageLease(PageLease&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)), handle_(other.handle_) {}

PageLease& PageLease::operator=(PageLease&& other) noexcept {
  if (this != &other) {
    reset();
    pool_ = std::exchange(other.pool_, nullptr);
    handle_ = other.handle_;
  }
  return *this;
}

std::optional<PageLease> PageLease::try_allocate(PagePool& pool) {
  const auto handle = pool.try_allocate();
  if (!handle) {
    return std::nullopt;
  }
  return PageLease{pool, *handle};
}

std::optional<PageLease> PageLease::try_share() const noexcept {
  if (pool_ == nullptr || pool_->retain(handle_) != RetainResult::Retained) {
    return std::nullopt;
  }
  // Construction and movement cannot throw after the successful retain.
  return PageLease{*pool_, handle_};
}

std::optional<PageHandle> PageLease::handle() const noexcept {
  return pool_ ? std::optional<PageHandle>{handle_} : std::nullopt;
}

void PageLease::reset() noexcept {
  if (pool_ != nullptr) {
    if (pool_->release(handle_) != ReleaseResult::Released) {
      std::terminate();
    }
    pool_ = nullptr;
  }
}

}  // namespace kvlab
