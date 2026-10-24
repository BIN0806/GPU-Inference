#include "kvlab/memory_accounting.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace kvlab {
namespace {
using Count = std::uint64_t;

Count add(Count a, Count b) {
  if (b > std::numeric_limits<Count>::max() - a) {
    throw std::overflow_error("memory accounting addition overflow");
  }
  return a + b;
}

Count multiply(Count a, Count b) {
  if (a != 0 && b > std::numeric_limits<Count>::max() / a) {
    throw std::overflow_error("memory accounting multiplication overflow");
  }
  return a * b;
}

struct PageUsage {
  Count references = 0, reserved = 0, committed = 0;
  PageHandle handle{};
  bool cached = false;
};
}  // namespace

MemorySnapshot account_memory(const PagePool& pool,
                              const std::vector<const RequestState*>& owners,
                              KvGeometry geometry, const RequestState* cache_owner) {
  static_assert(sizeof(std::size_t) <= sizeof(Count));
  if (!geometry.layers || !geometry.kv_heads || !geometry.head_dimension || !geometry.element_bytes) {
    throw std::invalid_argument("KV geometry dimensions must be positive");
  }
  MemorySnapshot result;
  result.bytes_per_token = multiply(multiply(multiply(multiply(2, geometry.layers),
                                                     geometry.kv_heads),
                                            geometry.head_dimension), geometry.element_bytes);
  const auto stats = pool.stats();
  const Count page_tokens = pool.tokens_per_page();
  result.pool_pages = stats.capacity;
  result.free_pages = stats.free_pages;
  result.live_pages = stats.allocated_pages;
  result.modeled_pool_bytes = multiply(multiply(result.pool_pages, page_tokens), result.bytes_per_token);
  result.live_capacity_tokens = multiply(result.live_pages, page_tokens);
  result.modeled_live_bytes = multiply(result.live_capacity_tokens, result.bytes_per_token);
  result.modeled_free_bytes = result.modeled_pool_bytes - result.modeled_live_bytes;

  pool.validate_invariants();
  if (cache_owner && std::find(owners.begin(), owners.end(), cache_owner) == owners.end()) {
    throw std::invalid_argument("cache owner must be included in accounting owners");
  }
  std::vector<PageUsage> usage(stats.capacity);
  std::set<const RequestState*> distinct;
  for (const auto* owner : owners) {
    if (!owner || !owner->belongs_to(pool) || !distinct.insert(owner).second) {
      throw std::invalid_argument("accounting needs distinct attached owners from this pool");
    }
    owner->validate_invariants();
    result.logical_page_references = add(result.logical_page_references, owner->page_count());
    result.logical_reserved_tokens = add(result.logical_reserved_tokens, owner->reserved_length());
    result.logical_committed_tokens = add(result.logical_committed_tokens, owner->length());
    Count reserved = owner->reserved_length(), committed = owner->length();
    for (std::size_t i = 0; i < owner->page_count(); ++i) {
      const auto handle = *owner->page_handle(i);  // Validated above against the issuing pool.
      auto& page = usage.at(handle.id);
      page.handle = handle;
      page.references = add(page.references, 1);
      page.cached = page.cached || owner == cache_owner;
      page.reserved = std::max(page.reserved, std::min(reserved, page_tokens));
      page.committed = std::max(page.committed, std::min(committed, page_tokens));
      reserved -= std::min(reserved, page_tokens);
      committed -= std::min(committed, page_tokens);
    }
  }
  Count seen = 0;
  for (const auto& page : usage) {
    if (!page.references) {
      continue;
    }
    const auto info = pool.inspect(page.handle);
    if (!info || info->references != page.references) {
      throw std::invalid_argument("accounting owners do not explain every page reference");
    }
    ++seen;
    if (page.cached && page.references == 1) ++result.cached_only_pages;
    result.unique_reserved_tokens = add(result.unique_reserved_tokens, page.reserved);
    result.unique_committed_tokens = add(result.unique_committed_tokens, page.committed);
  }
  if (seen != result.live_pages) {
    throw std::invalid_argument("accounting omitted a live page owner");
  }
  result.reserved_uncommitted_tokens = result.unique_reserved_tokens - result.unique_committed_tokens;
  result.tail_slack_tokens = result.live_capacity_tokens - result.unique_reserved_tokens;
  result.shared_capacity_saved_tokens =
      multiply(result.logical_page_references - result.live_pages, page_tokens);
  return result;
}

}  // namespace kvlab
