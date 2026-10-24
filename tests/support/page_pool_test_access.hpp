#pragma once

#include "kvlab/page_pool.hpp"

#include <stdexcept>

namespace kvlab::detail {

// Test-only seam for otherwise impractical counter/generation exhaustion cases.
// Restore injected metadata before teardown; never call this from production code.
struct PagePoolTestAccess {
  static void set_references(PagePool& pool, PageHandle handle, std::uint32_t count) {
    pool.slots_.at(handle.id).references = count;
  }
  static PageId next_free_id(const PagePool& pool, std::size_t offset) {
    return pool.free_ids_.at(pool.free_ids_.size() - 1 - offset);
  }
  static std::uint64_t generation(const PagePool& pool, PageId id) {
    return pool.slots_.at(id).generation;
  }
  static void set_free_generation(PagePool& pool, PageId id, std::uint64_t generation) {
    auto& slot = pool.slots_.at(id);
    if (slot.references != 0) {
      throw std::logic_error("Generation injection requires a free page");
    }
    slot.generation = generation;
  }
};

}  // namespace kvlab::detail
