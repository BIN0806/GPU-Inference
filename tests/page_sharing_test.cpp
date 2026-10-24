#include "kvlab/page_pool.hpp"
#include "kvlab/page_lease.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace kvlab {
namespace detail {
// Deliberate white-box seam for a boundary impractical to reach via billions of retains.
// The synthetic count is restored before teardown; this is not a real owner workload.
struct PagePoolTestAccess {
  static void set_references(PagePool& pool, PageHandle handle, std::uint32_t count) {
    pool.slots_.at(handle.id).references = count;
  }
};
}  // namespace detail
namespace {

TEST(PageSharing, PageGeometryMustBePositive) {
  EXPECT_THROW((PagePool{1, 0}), std::invalid_argument);
  EXPECT_THROW((PagePool{0, 0}), std::invalid_argument);
  EXPECT_EQ(PagePool(0).tokens_per_page(), 16U);
  EXPECT_EQ(PagePool(1, 4).tokens_per_page(), 4U);
}

TEST(PageSharing, AllocationStartsWithOneWritableReference) {
  PagePool pool(1);
  const auto handle = pool.try_allocate();
  ASSERT_TRUE(handle);
  const auto info = pool.inspect(*handle);
  ASSERT_TRUE(info);
  EXPECT_EQ(info->references, 1U);
  EXPECT_FALSE(info->sealed);
  EXPECT_TRUE(info->writable());
  EXPECT_EQ(pool.retain(*handle), RetainResult::NotSealed);
  EXPECT_EQ(pool.inspect(*handle)->references, 1U);
  EXPECT_EQ(pool.release(*handle), ReleaseResult::Released);
}

TEST(PageSharing, OnlyExactlyFullPagesCanBeSealed) {
  PagePool pool(1, 4);
  const auto handle = pool.try_allocate();
  ASSERT_TRUE(handle);
  for (const auto count : std::array<std::size_t, 4>{0, 1, 3, 5}) {
    EXPECT_EQ(pool.seal_full(*handle, count), SealResult::NotFull);
    EXPECT_TRUE(pool.inspect(*handle)->writable());
    EXPECT_EQ(pool.retain(*handle), RetainResult::NotSealed);
  }
  EXPECT_EQ(pool.seal_full(*handle, 4), SealResult::Sealed);
  EXPECT_FALSE(pool.inspect(*handle)->writable());
  EXPECT_EQ(pool.seal_full(*handle, 4), SealResult::AlreadySealed);
  EXPECT_EQ(pool.inspect(*handle)->references, 1U);
  EXPECT_EQ(pool.release(*handle), ReleaseResult::Released);
  EXPECT_NO_THROW(pool.validate_invariants());
}

TEST(PageSharing, OnlyFinalReleaseReturnsPhysicalCapacity) {
  PagePool pool(1, 4);
  const auto handle = pool.try_allocate();
  ASSERT_TRUE(handle);
  ASSERT_EQ(pool.seal_full(*handle, 4), SealResult::Sealed);
  ASSERT_EQ(pool.retain(*handle), RetainResult::Retained);
  ASSERT_EQ(pool.retain(*handle), RetainResult::Retained);
  EXPECT_EQ(pool.inspect(*handle)->references, 3U);
  EXPECT_EQ(pool.stats().allocated_pages, 1U);
  for (const auto remaining : {2U, 1U}) {
    ASSERT_EQ(pool.release(*handle), ReleaseResult::Released);
    EXPECT_EQ(pool.inspect(*handle)->references, remaining);
    EXPECT_TRUE(pool.inspect(*handle)->sealed);
    EXPECT_FALSE(pool.inspect(*handle)->writable());
    EXPECT_EQ(pool.stats().free_pages, 0U);
    EXPECT_FALSE(pool.try_allocate());
    EXPECT_NO_THROW(pool.validate_invariants());
  }
  EXPECT_EQ(pool.release(*handle), ReleaseResult::Released);
  EXPECT_EQ(pool.stats().free_pages, 1U);
  EXPECT_FALSE(pool.inspect(*handle));
  EXPECT_EQ(pool.release(*handle), ReleaseResult::NotAllocated);
  EXPECT_NO_THROW(pool.validate_invariants());
}

TEST(PageSharing, ReallocationResetsSealAndInvalidatesEveryOldOperation) {
  PagePool pool(1, 4);
  const auto old = pool.try_allocate();
  ASSERT_TRUE(old);
  ASSERT_EQ(pool.seal_full(*old, 4), SealResult::Sealed);
  ASSERT_EQ(pool.retain(*old), RetainResult::Retained);
  ASSERT_EQ(pool.release(*old), ReleaseResult::Released);
  ASSERT_EQ(pool.release(*old), ReleaseResult::Released);
  const auto current = pool.try_allocate();
  ASSERT_TRUE(current);
  EXPECT_EQ(old->id, current->id);
  EXPECT_GT(current->generation, old->generation);
  EXPECT_TRUE(pool.inspect(*current)->writable());
  EXPECT_EQ(pool.inspect(*current)->references, 1U);
  EXPECT_EQ(pool.retain(*old), RetainResult::StaleHandle);
  EXPECT_EQ(pool.seal_full(*old, 4), SealResult::StaleHandle);
  EXPECT_EQ(pool.release(*old), ReleaseResult::StaleHandle);
  EXPECT_FALSE(pool.inspect(*old));
  EXPECT_TRUE(pool.inspect(*current)->writable());
  EXPECT_EQ(pool.release(*current), ReleaseResult::Released);
}

TEST(PageSharing, InvalidAndFreeHandlesCannotRetainOrSeal) {
  PagePool pool(1, 4);
  const PageHandle invalid{std::numeric_limits<PageId>::max(), 1};
  EXPECT_EQ(pool.retain(invalid), RetainResult::InvalidPage);
  EXPECT_EQ(pool.seal_full(invalid, 4), SealResult::InvalidPage);
  EXPECT_FALSE(pool.inspect(invalid));
  EXPECT_EQ(pool.retain({0, 1}), RetainResult::NotAllocated);
  EXPECT_EQ(pool.seal_full({0, 1}, 4), SealResult::NotAllocated);
  EXPECT_FALSE(pool.inspect({0, 1}));
  EXPECT_EQ(pool.stats().free_pages, 1U);
  const auto live = pool.try_allocate();
  ASSERT_TRUE(live);
  const PageHandle wrong{live->id, live->generation + 1};
  EXPECT_EQ(pool.retain(wrong), RetainResult::StaleHandle);
  EXPECT_EQ(pool.seal_full(wrong, 4), SealResult::StaleHandle);
  EXPECT_FALSE(pool.inspect(wrong));
  EXPECT_TRUE(pool.inspect(*live)->writable());
  EXPECT_EQ(pool.release(*live), ReleaseResult::Released);
}

TEST(PageSharing, RetainOverflowLeavesCounterUnchanged) {
  PagePool pool(1, 4);
  const auto handle = pool.try_allocate();
  ASSERT_TRUE(handle);
  ASSERT_EQ(pool.seal_full(*handle, 4), SealResult::Sealed);
  constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
  detail::PagePoolTestAccess::set_references(pool, *handle, maximum - 1);
  EXPECT_EQ(pool.retain(*handle), RetainResult::Retained);
  EXPECT_EQ(pool.inspect(*handle)->references, maximum);
  EXPECT_EQ(pool.retain(*handle), RetainResult::Overflow);
  EXPECT_EQ(pool.inspect(*handle)->references, maximum);
  EXPECT_EQ(pool.stats().free_pages, 0U);
  EXPECT_EQ(pool.release(*handle), ReleaseResult::Released);
  EXPECT_EQ(pool.inspect(*handle)->references, maximum - 1);
  EXPECT_NO_THROW(pool.validate_invariants());
  detail::PagePoolTestAccess::set_references(pool, *handle, 1);
  EXPECT_EQ(pool.release(*handle), ReleaseResult::Released);
  EXPECT_EQ(pool.stats().free_pages, 1U);
}

TEST(PageSharing, LeaseShareOverflowCreatesNoOwner) {
  PagePool pool(1, 4);
  auto owner = PageLease::try_allocate(pool);
  ASSERT_TRUE(owner);
  const auto handle = owner->handle();
  ASSERT_TRUE(handle);
  ASSERT_EQ(pool.seal_full(*handle, 4), SealResult::Sealed);
  constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
  detail::PagePoolTestAccess::set_references(pool, *handle, maximum);
  EXPECT_FALSE(owner->try_share());
  EXPECT_EQ(pool.inspect(*handle)->references, maximum);
  detail::PagePoolTestAccess::set_references(pool, *handle, 1);
  owner->reset();
  EXPECT_EQ(pool.stats().free_pages, 1U);
}

TEST(PageSharing, SeededOwnerListsAgreeWithReferenceCounts) {
  constexpr std::size_t capacity = 7;
  constexpr std::size_t page_tokens = 4;
  for (std::uint32_t seed = 0; seed < 100; ++seed) {
    SCOPED_TRACE(::testing::Message() << "seed=" << seed);
    PagePool pool(capacity, page_tokens);
    std::set<PageId> free_ids{0, 1, 2, 3, 4, 5, 6};
    std::set<PageId> sealed;
    // Each entry models a distinct owner. Counts are derived from this list, not mirrored.
    std::vector<PageHandle> owners;
    std::array<std::uint64_t, capacity> generations{};
    std::mt19937 random(seed);
    for (int step = 0; step < 10000; ++step) {
      SCOPED_TRACE(::testing::Message() << "step=" << step);
      const auto operation = random() % 4;
      if (operation == 0 || owners.empty()) {
        const auto handle = pool.try_allocate();
        ASSERT_EQ(handle.has_value(), !free_ids.empty());
        if (handle) {
          ASSERT_LT(handle->id, capacity);
          ASSERT_EQ(free_ids.erase(handle->id), 1U);
          EXPECT_GT(handle->generation, generations[handle->id]);
          generations[handle->id] = handle->generation;
          owners.push_back(*handle);
        }
      } else {
        const auto index = static_cast<std::size_t>(random() % owners.size());
        const auto handle = owners[index];
        if (operation == 1 && owners.size() < 32) {
          const auto result = pool.retain(handle);
          if (sealed.count(handle.id) != 0) {
            ASSERT_EQ(result, RetainResult::Retained);
            owners.push_back(handle);
          } else {
            ASSERT_EQ(result, RetainResult::NotSealed);
          }
        } else if (operation == 2) {
          const std::size_t valid_tokens = page_tokens - 1 + random() % 3;
          const auto result = pool.seal_full(handle, valid_tokens);
          if (sealed.count(handle.id) != 0) {
            ASSERT_EQ(result, SealResult::AlreadySealed);
          } else if (valid_tokens == page_tokens) {
            ASSERT_EQ(result, SealResult::Sealed);
            sealed.insert(handle.id);
          } else {
            ASSERT_EQ(result, SealResult::NotFull);
          }
        } else {
          ASSERT_EQ(pool.release(handle), ReleaseResult::Released);
          owners.erase(owners.begin() + static_cast<std::ptrdiff_t>(index));
          const bool still_owned = std::any_of(owners.begin(), owners.end(), [handle](auto owner) {
            return owner.id == handle.id;
          });
          if (!still_owned) {
            free_ids.insert(handle.id);
            sealed.erase(handle.id);
            EXPECT_EQ(pool.release(handle), ReleaseResult::NotAllocated);
          }
        }
      }
      EXPECT_EQ(pool.stats().free_pages, free_ids.size());
      EXPECT_EQ(pool.stats().allocated_pages, capacity - free_ids.size());
      for (PageId id = 0; id < capacity; ++id) {
        const auto count = std::count_if(owners.begin(), owners.end(), [id](auto owner) {
          return owner.id == id;
        });
        const auto info = pool.inspect({id, generations[id]});
        ASSERT_EQ(info.has_value(), count != 0);
        if (info) {
          EXPECT_EQ(info->references, static_cast<std::uint32_t>(count));
          EXPECT_EQ(info->sealed, sealed.count(id) != 0);
          EXPECT_EQ(info->writable(), count == 1 && sealed.count(id) == 0);
        }
      }
      if (step % 100 == 0) {
        ASSERT_NO_THROW(pool.validate_invariants());
      }
    }
    for (const auto handle : owners) {
      ASSERT_EQ(pool.release(handle), ReleaseResult::Released);
    }
    EXPECT_EQ(pool.stats().free_pages, capacity);
    EXPECT_NO_THROW(pool.validate_invariants());
  }
}

}  // namespace
}  // namespace kvlab
