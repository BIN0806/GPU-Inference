#pragma once

#include "kvlab/page_pool.hpp"

#include <optional>

namespace kvlab {

// Owns exactly one pool reference. The issuing pool MUST outlive all its leases.
// Never manually retain/release a borrowed handle on behalf of this lease.
// Single-threaded; destruction does not wait for GPU completion.
class PageLease {
 public:
  PageLease() noexcept = default;
  ~PageLease() noexcept;
  PageLease(const PageLease&) = delete;
  PageLease& operator=(const PageLease&) = delete;
  PageLease(PageLease&& other) noexcept;
  PageLease& operator=(PageLease&& other) noexcept;

  // Pool exhaustion returns nullopt; pool construction/generation exceptions are unchanged.
  [[nodiscard]] static std::optional<PageLease> try_allocate(PagePool& pool);
  // No change on failure (empty lease, unsealed page, invalid handle, or count overflow).
  [[nodiscard]] std::optional<PageLease> try_share() const noexcept;
  // A borrowed token: copying it does not retain. State queries/sealing may use it.
  [[nodiscard]] std::optional<PageHandle> handle() const noexcept;
  explicit operator bool() const noexcept { return pool_ != nullptr; }
  // Idempotent for an empty lease. An invalid underlying release terminates:
  // that signals a violated ownership contract, never a normal capacity failure.
  void reset() noexcept;

 private:
  PageLease(PagePool& pool, PageHandle handle) noexcept : pool_(&pool), handle_(handle) {}
  PagePool* pool_ = nullptr;
  PageHandle handle_{};
};

}  // namespace kvlab
