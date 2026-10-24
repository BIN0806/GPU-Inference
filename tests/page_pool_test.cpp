#include "kvlab/page_pool.hpp"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace kvlab {
namespace {

static_assert(!std::is_copy_constructible_v<PagePool>);
static_assert(!std::is_move_constructible_v<PagePool>);

TEST(PagePool, ZeroCapacityIsExhausted) {
  PagePool pool(0);
  EXPECT_FALSE(pool.try_allocate());
  EXPECT_EQ(pool.stats().capacity, 0U);
  EXPECT_EQ(pool.stats().free_pages, 0U);
  EXPECT_EQ(pool.release({0, 1}), ReleaseResult::InvalidPage);
  EXPECT_NO_THROW(pool.validate_invariants());
}

TEST(PagePool, FourUniquePagesThenExhaustion) {
  PagePool pool(4);
  std::set<PageId> ids;
  for (int i = 0; i < 4; ++i) {
    const auto handle = pool.try_allocate();
    ASSERT_TRUE(handle);
    EXPECT_LT(handle->id, 4U);
    EXPECT_TRUE(ids.insert(handle->id).second);
  }
  EXPECT_FALSE(pool.try_allocate());
  EXPECT_FALSE(pool.try_allocate());
  EXPECT_EQ(pool.stats().allocated_pages, 4U);
  EXPECT_EQ(pool.stats().free_pages, 0U);
  EXPECT_NO_THROW(pool.validate_invariants());
}

TEST(PagePool, ReusesPageWithNewGeneration) {
  PagePool pool(1);
  const auto first = pool.try_allocate();
  ASSERT_TRUE(first);
  EXPECT_EQ(pool.release(*first), ReleaseResult::Released);
  const auto second = pool.try_allocate();
  ASSERT_TRUE(second);
  EXPECT_EQ(first->id, second->id);
  EXPECT_GT(second->generation, first->generation);
  EXPECT_NO_THROW(pool.validate_invariants());
}

TEST(PagePool, RejectsDuplicateReleaseWithoutChangingCapacity) {
  PagePool pool(1);
  const auto handle = pool.try_allocate();
  ASSERT_TRUE(handle);
  EXPECT_EQ(pool.release(*handle), ReleaseResult::Released);
  EXPECT_EQ(pool.release(*handle), ReleaseResult::NotAllocated);
  EXPECT_EQ(pool.stats().free_pages, 1U);
  ASSERT_TRUE(pool.try_allocate());
  EXPECT_FALSE(pool.try_allocate());
  EXPECT_NO_THROW(pool.validate_invariants());
}

TEST(PagePool, RejectsOutOfRangePageWithoutChangingLiveState) {
  PagePool pool(1);
  const auto live = pool.try_allocate();
  ASSERT_TRUE(live);
  EXPECT_EQ(pool.release({1, live->generation}), ReleaseResult::InvalidPage);
  EXPECT_EQ(pool.release({std::numeric_limits<PageId>::max(), 1}), ReleaseResult::InvalidPage);
  EXPECT_EQ(pool.stats().allocated_pages, 1U);
  EXPECT_EQ(pool.release(*live), ReleaseResult::Released);
  EXPECT_NO_THROW(pool.validate_invariants());
}

TEST(PagePool, OldHandleCannotReleaseANewAllocation) {
  PagePool pool(1);
  const auto old = pool.try_allocate();
  ASSERT_TRUE(old);
  ASSERT_EQ(pool.release(*old), ReleaseResult::Released);
  const auto current = pool.try_allocate();
  ASSERT_TRUE(current);
  EXPECT_EQ(pool.release(*old), ReleaseResult::StaleHandle);
  EXPECT_EQ(pool.stats().allocated_pages, 1U);
  EXPECT_FALSE(pool.try_allocate());
  EXPECT_EQ(pool.release(*current), ReleaseResult::Released);
  EXPECT_NO_THROW(pool.validate_invariants());
}

TEST(PagePool, RejectsIncorrectGenerationWithoutChangingLiveState) {
  PagePool pool(1);
  const auto live = pool.try_allocate();
  ASSERT_TRUE(live);
  EXPECT_EQ(pool.release({live->id, 0}), ReleaseResult::StaleHandle);
  EXPECT_EQ(pool.release({live->id, live->generation + 1}), ReleaseResult::StaleHandle);
  EXPECT_EQ(pool.stats().allocated_pages, 1U);
  EXPECT_EQ(pool.release(*live), ReleaseResult::Released);
}

TEST(PagePool, RejectsNeverAllocatedPage) {
  PagePool pool(2);
  EXPECT_EQ(pool.release({1, 1}), ReleaseResult::NotAllocated);
  EXPECT_EQ(pool.stats().free_pages, 2U);
  EXPECT_NO_THROW(pool.validate_invariants());
}

TEST(PagePool, ArbitraryReleaseOrderRestoresAllPages) {
  PagePool pool(8);
  std::vector<PageHandle> handles;
  for (int i = 0; i < 8; ++i) {
    const auto handle = pool.try_allocate();
    ASSERT_TRUE(handle);
    handles.push_back(*handle);
  }
  for (const auto i : std::array<std::size_t, 8>{3, 0, 6, 1, 7, 4, 2, 5}) {
    EXPECT_EQ(pool.release(handles[i]), ReleaseResult::Released);
    EXPECT_NO_THROW(pool.validate_invariants());
  }
  EXPECT_EQ(pool.stats().free_pages, 8U);
  EXPECT_EQ(pool.stats().allocated_pages, 0U);
  std::set<PageId> second_round;
  for (int i = 0; i < 8; ++i) {
    const auto handle = pool.try_allocate();
    ASSERT_TRUE(handle);
    EXPECT_TRUE(second_round.insert(handle->id).second);
  }
}

TEST(PagePool, RejectsUnrepresentableCapacityBeforeAllocatingStorage) {
  EXPECT_THROW(PagePool{PagePool::max_capacity() + 1}, std::length_error);
}

TEST(PagePool, SeededOperationsAgreeWithIndependentSetModel) {
  constexpr std::size_t capacity = 17;
  for (std::uint32_t seed = 0; seed < 16; ++seed) {
    PagePool pool(capacity);
    std::set<PageId> free_model;
    for (PageId id = 0; id < capacity; ++id) {
      free_model.insert(id);
    }
    std::vector<PageHandle> live;
    std::array<std::uint64_t, capacity> previous_generations{};
    std::mt19937 random(seed);
    for (int step = 0; step < 10000; ++step) {
      SCOPED_TRACE(::testing::Message() << "seed=" << seed << ", step=" << step);
      if (random() % 2 == 0 || live.empty()) {
        const auto handle = pool.try_allocate();
        ASSERT_EQ(handle.has_value(), !free_model.empty());
        if (handle) {
          ASSERT_LT(handle->id, capacity);
          ASSERT_EQ(free_model.erase(handle->id), 1U);
          EXPECT_GT(handle->generation, previous_generations[handle->id]);
          previous_generations[handle->id] = handle->generation;
          live.push_back(*handle);
        }
      } else {
        const auto index = static_cast<std::size_t>(random() % live.size());
        const auto handle = live[index];
        ASSERT_EQ(pool.release(handle), ReleaseResult::Released);
        EXPECT_TRUE(free_model.insert(handle.id).second);
        EXPECT_EQ(pool.release(handle), ReleaseResult::NotAllocated);
        live[index] = live.back();
        live.pop_back();
      }
      EXPECT_EQ(pool.stats().free_pages, free_model.size());
      EXPECT_EQ(pool.stats().allocated_pages, live.size());
      if (step % 100 == 0) {
        ASSERT_NO_THROW(pool.validate_invariants());
      }
    }
    for (const auto handle : live) {
      ASSERT_EQ(pool.release(handle), ReleaseResult::Released);
    }
    EXPECT_EQ(pool.stats().free_pages, capacity);
    EXPECT_NO_THROW(pool.validate_invariants());
  }
}

}  // namespace
}  // namespace kvlab
