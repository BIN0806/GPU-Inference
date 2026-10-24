#include "kvlab/page_pool.hpp"

#include <stdexcept>

namespace kvlab {

PagePool::PagePool(std::size_t capacity) {
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
  slot.allocated = true;
  return PageHandle{id, slot.generation};
}

ReleaseResult PagePool::release(PageHandle handle) noexcept {
  if (handle.id >= slots_.size()) {
    return ReleaseResult::InvalidPage;
  }
  auto& slot = slots_[handle.id];
  if (!slot.allocated) {
    return ReleaseResult::NotAllocated;
  }
  if (slot.generation != handle.generation) {
    return ReleaseResult::StaleHandle;
  }
  // Construction reserved capacity entries. A live page has no entry in this stack,
  // so returning it fits without allocation; PageId copying cannot throw.
  free_ids_.push_back(handle.id);
  slot.allocated = false;
  return ReleaseResult::Released;
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
    if (id >= slots_.size() || seen[id] != 0 || slots_[id].allocated) {
      throw std::logic_error("PagePool free list contains an invalid, duplicate, or live ID");
    }
    seen[id] = 1;
  }
  for (std::size_t id = 0; id < slots_.size(); ++id) {
    if (slots_[id].allocated) {
      if (seen[id] != 0 || slots_[id].generation == 0) {
        throw std::logic_error("PagePool live-page invariant failed");
      }
    } else if (seen[id] != 1) {
      throw std::logic_error("PagePool free page is missing from the stack");
    }
  }
}

}  // namespace kvlab
