#pragma once

#include "kvlab/page_lease.hpp"

#include <cstddef>
#include <memory_resource>
#include <optional>
#include <vector>

namespace kvlab {

enum class RequestStatus {
  Ok, PoolExhausted, InvalidLength, NotEmpty, PoolMismatch, NotShareable, NotWritable, MovedFrom
};

struct TokenLocation {
  PageHandle page;
  std::size_t offset;
};

// CPU metadata only, single thread. Pool and memory resource must outlive this object.
// Borrowed handles must not be manually released, retained, or sealed by the caller.
class RequestState {
 public:
  explicit RequestState(PagePool& pool,
                        std::pmr::memory_resource* resource = std::pmr::get_default_resource());
  ~RequestState() = default;
  RequestState(const RequestState&) = delete;
  RequestState& operator=(const RequestState&) = delete;
  RequestState(RequestState&& other) noexcept;
  // Keeps destination's memory resource. May throw before changing either request.
  RequestState& operator=(RequestState&& other);

  // Monotonic capacity reservation; target <= reserved_length is a no-op.
  // Does not commit tokens. bad_alloc/length_error/generation overflow may propagate.
  // On failure, existing handles, lengths, and ownership are unchanged. Briefly
  // acquired/free pages may have advanced generations or a different free-list order.
  // Host table capacity may grow even when later page acquisition fails; it is
  // retained across reset for reuse. A metadata allocation failure has no effects.
  [[nodiscard]] RequestStatus reserve_for_length(std::size_t target);
  // Caller attests data through target has been materialized; does not write payloads
  // or wait for a GPU. Rejects shrinking or exceeding the reserved length.
  [[nodiscard]] RequestStatus commit_length(std::size_t target) noexcept;
  // Destination must be empty, pools identical, source prefix committed and complete.
  // On retain failure or host exception, earlier temporary retains are rolled back.
  [[nodiscard]] RequestStatus share_prefix_from(const RequestState& source,
                                                std::size_t full_pages);

  void reset() noexcept;
  std::size_t length() const noexcept { return length_; }
  std::size_t reserved_length() const noexcept { return reserved_length_; }
  std::size_t page_count() const noexcept { return pages_.size(); }
  bool attached() const noexcept { return pool_ != nullptr; }
  bool belongs_to(const PagePool& pool) const noexcept { return pool_ == &pool; }
  // Includes reserved pages; a token location is exposed only for committed positions.
  [[nodiscard]] std::optional<PageHandle> page_handle(std::size_t logical_page) const noexcept;
  [[nodiscard]] std::optional<TokenLocation> token_location(std::size_t position) const noexcept;
  // Diagnostic O(page count) scan with temporary storage, never a hot-path operation.
  void validate_invariants() const;

 private:
  using Table = std::pmr::vector<PageLease>;
  PagePool* pool_;
  Table pages_;
  std::size_t length_ = 0;
  std::size_t reserved_length_ = 0;
};

}  // namespace kvlab
