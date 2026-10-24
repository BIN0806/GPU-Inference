#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace kvlab {

using PageId = std::uint32_t;

// Scoped to the pool that issued it. A handle is a token, not an RAII owner.
// Do not pass it to another pool or use it after its pool has been destroyed.
struct PageHandle {
  PageId id;
  std::uint64_t generation;
};

struct PoolStats {
  std::size_t capacity;
  std::size_t free_pages;
  std::size_t allocated_pages;
};

enum class ReleaseResult { Released, InvalidPage, NotAllocated, StaleHandle };

// Single-threaded, CPU metadata only. Owns no KV payload and no GPU resources.
class PagePool {
 public:
  explicit PagePool(std::size_t capacity);
  PagePool(const PagePool&) = delete;
  PagePool& operator=(const PagePool&) = delete;
  PagePool(PagePool&&) = delete;
  PagePool& operator=(PagePool&&) = delete;

  // Keep IDs representable in the planned signed int32 GPU table (-1 is invalid).
  static constexpr std::size_t max_capacity() noexcept {
    return static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  }

  // Exhaustion returns nullopt. Generation exhaustion throws before changing state.
  [[nodiscard]] std::optional<PageHandle> try_allocate();
  // Invalid releases leave state unchanged. Valid releases cannot grow the free list's capacity.
  [[nodiscard]] ReleaseResult release(PageHandle handle) noexcept;
  [[nodiscard]] PoolStats stats() const noexcept;

  // O(capacity) diagnostic with temporary storage; never part of a hot-path timing.
  // Throws logic_error on an invariant violation (bad_alloc can propagate as well).
  void validate_invariants() const;

 private:
  struct Slot {
    std::uint64_t generation = 0;
    bool allocated = false;
  };

  std::vector<Slot> slots_;
  std::vector<PageId> free_ids_;
};

}  // namespace kvlab
