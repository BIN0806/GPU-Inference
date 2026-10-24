#pragma once

#include "kvlab/request_state.hpp"

#include <cstdint>
#include <vector>

namespace kvlab {

struct KvGeometry {
  std::uint64_t layers = 1;
  std::uint64_t kv_heads = 8;
  std::uint64_t head_dimension = 64;
  std::uint64_t element_bytes = 2;
};

struct MemorySnapshot {
  std::uint64_t pool_pages = 0, free_pages = 0, live_pages = 0;
  std::uint64_t cached_only_pages = 0;
  std::uint64_t logical_page_references = 0;
  std::uint64_t logical_committed_tokens = 0, logical_reserved_tokens = 0;
  std::uint64_t unique_committed_tokens = 0, unique_reserved_tokens = 0;
  std::uint64_t live_capacity_tokens = 0;
  std::uint64_t reserved_uncommitted_tokens = 0, tail_slack_tokens = 0;
  std::uint64_t shared_capacity_saved_tokens = 0;
  std::uint64_t bytes_per_token = 0;
  std::uint64_t modeled_pool_bytes = 0, modeled_live_bytes = 0, modeled_free_bytes = 0;
};

// Diagnostic only: O(pool capacity + logical page references) storage/work, plus
// request invariant checks. Every live reference must belong to exactly one
// supplied request (including an explicit prefix owner). No concurrent mutation.
// Rejects null/duplicate/foreign/moved-from owners or incomplete ownership.
// Checked arithmetic throws overflow_error; invalid inputs throw invalid_argument.
// Optional cache_owner must be one of owners; pages it alone references are
// reported separately. Supports one explicit prefix owner, not a general cache.
// Bytes are a hypothetical KV layout model, not allocated payload or process RSS.
MemorySnapshot account_memory(const PagePool& pool,
                              const std::vector<const RequestState*>& owners,
                              KvGeometry geometry = {}, const RequestState* cache_owner = nullptr);

}  // namespace kvlab
