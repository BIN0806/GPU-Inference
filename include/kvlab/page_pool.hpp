#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace kvlab {

using PageId = std::uint32_t;

namespace detail {
struct PagePoolTestAccess;  // Defined only by tests to exercise counter exhaustion.
}

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

struct PageInfo {
  std::uint32_t references;
  bool sealed;
  // A metadata permission, not enforcement of writes to a future payload buffer.
  bool writable() const noexcept { return references == 1 && !sealed; }
};

enum class ReleaseResult { Released, InvalidPage, NotAllocated, StaleHandle };
enum class RetainResult { Retained, InvalidPage, NotAllocated, StaleHandle, NotSealed, Overflow };
enum class SealResult { Sealed, InvalidPage, NotAllocated, StaleHandle, NotFull, AlreadySealed };

// Single-threaded, CPU metadata only. Owns no KV payload and no GPU resources.
class PagePool {
 public:
  explicit PagePool(std::size_t capacity, std::size_t tokens_per_page = 16);
  PagePool(const PagePool&) = delete;
  PagePool& operator=(const PagePool&) = delete;
  PagePool(PagePool&&) = delete;
  PagePool& operator=(PagePool&&) = delete;

  // Keep IDs representable in the planned signed int32 GPU table (-1 is invalid).
  static constexpr std::size_t max_capacity() noexcept {
    return static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  }

  // Starts with one reference. Exhaustion returns nullopt; generation exhaustion throws.
  [[nodiscard]] std::optional<PageHandle> try_allocate();
  // Caller attests that valid_tokens have been materialized. Only a full page can seal.
  // Sealing is irreversible until final release; this pool owns no payload to inspect.
  [[nodiscard]] SealResult seal_full(PageHandle handle, std::size_t valid_tokens) noexcept;
  // One successful retain creates one obligation to release. Only sealed pages may share.
  [[nodiscard]] RetainResult retain(PageHandle handle) noexcept;
  // Released means ONE reference dropped; the page becomes free only at zero.
  // The caller must own that reference. Handles do not identify individual owners;
  // a duplicate release while another reference exists cannot be detected here.
  [[nodiscard]] ReleaseResult release(PageHandle handle) noexcept;
  [[nodiscard]] std::optional<PageInfo> inspect(PageHandle handle) const noexcept;
  [[nodiscard]] PoolStats stats() const noexcept;
  std::size_t tokens_per_page() const noexcept { return tokens_per_page_; }

  // O(capacity) diagnostic with temporary storage; never part of a hot-path timing.
  // Throws logic_error on an invariant violation (bad_alloc can propagate as well).
  void validate_invariants() const;

 private:
  friend struct detail::PagePoolTestAccess;
  struct Slot {
    std::uint64_t generation = 0;
    std::uint32_t references = 0;
    bool sealed = false;
  };

  std::size_t tokens_per_page_;
  std::vector<Slot> slots_;
  std::vector<PageId> free_ids_;
};

}  // namespace kvlab
