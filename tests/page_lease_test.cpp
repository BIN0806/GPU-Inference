#include "kvlab/page_lease.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace kvlab {
namespace {

static_assert(!std::is_copy_constructible_v<PageLease>);
static_assert(!std::is_copy_assignable_v<PageLease>);
static_assert(std::is_nothrow_move_constructible_v<PageLease>);
static_assert(std::is_nothrow_move_assignable_v<PageLease>);
static_assert(std::is_nothrow_destructible_v<PageLease>);

TEST(PageLease, EmptyLeaseIsSafeToResetAndMove) {
  kvlab::PageLease empty;
  EXPECT_FALSE(empty);
  EXPECT_FALSE(empty.handle());
  EXPECT_FALSE(empty.try_share());
  empty.reset();
  kvlab::PageLease moved(std::move(empty));
  EXPECT_FALSE(moved);
  EXPECT_FALSE(empty);
}

TEST(PageLease, ExhaustionAndDestructionRestoreCapacity) {
  PagePool pool(1);
  {
    const auto owner = kvlab::PageLease::try_allocate(pool);
    ASSERT_TRUE(owner);
    EXPECT_FALSE(kvlab::PageLease::try_allocate(pool));
    EXPECT_EQ(pool.stats().allocated_pages, 1U);
  }
  EXPECT_EQ(pool.stats().free_pages, 1U);
  EXPECT_NO_THROW(pool.validate_invariants());
}

TEST(PageLease, SharingMutablePageFailsWithoutAddingOwner) {
  PagePool pool(1, 4);
  auto owner = kvlab::PageLease::try_allocate(pool);
  ASSERT_TRUE(owner);
  EXPECT_FALSE(owner->try_share());
  EXPECT_EQ(pool.inspect(*owner->handle())->references, 1U);
  EXPECT_EQ(pool.stats().free_pages, 0U);
}

TEST(PageLease, MoveTransfersReferenceWithoutRetaining) {
  PagePool pool(1);
  auto original = kvlab::PageLease::try_allocate(pool);
  ASSERT_TRUE(original);
  const auto handle = original->handle();
  ASSERT_TRUE(handle);
  kvlab::PageLease moved(std::move(*original));
  EXPECT_FALSE(*original);
  EXPECT_FALSE(original->handle());
  EXPECT_FALSE(original->try_share());
  original->reset();
  EXPECT_EQ(pool.inspect(*handle)->references, 1U);
  moved.reset();
  moved.reset();
  EXPECT_EQ(pool.stats().free_pages, 1U);
}

TEST(PageLease, MoveAssignmentReleasesPreviousDestinationAcrossPools) {
  PagePool first(1), second(1);
  auto destination = kvlab::PageLease::try_allocate(first);
  auto source = kvlab::PageLease::try_allocate(second);
  ASSERT_TRUE(destination);
  ASSERT_TRUE(source);
  *destination = std::move(*source);
  EXPECT_EQ(first.stats().free_pages, 1U);
  EXPECT_EQ(second.stats().free_pages, 0U);
  EXPECT_FALSE(*source);
  destination->reset();
  EXPECT_EQ(second.stats().free_pages, 1U);
}

TEST(PageLease, MoveAssignmentBetweenSharedOwnersKeepsOneReference) {
  PagePool pool(1, 4);
  auto first = kvlab::PageLease::try_allocate(pool);
  ASSERT_TRUE(first);
  const auto handle = first->handle();
  ASSERT_TRUE(handle);
  ASSERT_EQ(pool.seal_full(*handle, 4), SealResult::Sealed);
  auto second = first->try_share();
  ASSERT_TRUE(second);
  EXPECT_EQ(second->handle()->id, handle->id);
  EXPECT_EQ(second->handle()->generation, handle->generation);
  *first = std::move(*second);
  EXPECT_FALSE(*second);
  EXPECT_EQ(pool.inspect(*handle)->references, 1U);
  EXPECT_TRUE(pool.inspect(*handle)->sealed);
  first->reset();
  EXPECT_EQ(pool.stats().free_pages, 1U);
}

TEST(PageLease, SelfMoveLeavesOwnershipIntact) {
  PagePool pool(1);
  auto owner = kvlab::PageLease::try_allocate(pool);
  ASSERT_TRUE(owner);
  auto* alias = &*owner;
  *owner = std::move(*alias);
  EXPECT_TRUE(*owner);
  EXPECT_EQ(pool.inspect(*owner->handle())->references, 1U);
}

TEST(PageLease, ThreeSharedOwnersCanFinishInEveryOrder) {
  std::array<std::size_t, 3> order{0, 1, 2};
  do {
    PagePool pool(1, 4);
    std::array<kvlab::PageLease, 3> owners;
    auto first = kvlab::PageLease::try_allocate(pool);
    ASSERT_TRUE(first);
    owners[0] = std::move(*first);
    const auto handle = owners[0].handle();
    ASSERT_TRUE(handle);
    ASSERT_EQ(pool.seal_full(*handle, 4), SealResult::Sealed);
    for (std::size_t i = 1; i < owners.size(); ++i) {
      auto copy = owners[0].try_share();
      ASSERT_TRUE(copy);
      owners[i] = std::move(*copy);
    }
    for (std::size_t i = 0; i < order.size(); ++i) {
      owners[order[i]].reset();
      owners[order[i]].reset();
      if (i < 2) {
        EXPECT_EQ(pool.stats().free_pages, 0U);
        ASSERT_TRUE(pool.inspect(*handle));
        EXPECT_EQ(pool.inspect(*handle)->references, 2U - i);
      } else {
        EXPECT_EQ(pool.stats().free_pages, 1U);
        EXPECT_FALSE(pool.inspect(*handle));
      }
      EXPECT_NO_THROW(pool.validate_invariants());
    }
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST(PageLease, ExceptionUnwindingReleasesOnlyNewOwners) {
  PagePool pool(3, 4);
  auto prefix = kvlab::PageLease::try_allocate(pool);
  ASSERT_TRUE(prefix);
  const auto handle = prefix->handle();
  ASSERT_TRUE(handle);
  ASSERT_EQ(pool.seal_full(*handle, 4), SealResult::Sealed);
  const auto fail = [&] {
    auto shared = prefix->try_share();
    auto tail = kvlab::PageLease::try_allocate(pool);
    if (!shared || !tail) {
      throw std::logic_error("unexpected capacity failure");
    }
    throw std::runtime_error("synthetic request cancellation");
  };
  EXPECT_THROW(fail(), std::runtime_error);
  EXPECT_EQ(pool.inspect(*handle)->references, 1U);
  EXPECT_EQ(pool.stats().allocated_pages, 1U);
  EXPECT_EQ(pool.stats().free_pages, 2U);
  prefix->reset();
  EXPECT_EQ(pool.stats().free_pages, 3U);
}

TEST(PageLease, VectorRelocationAndBorrowedHandleCopiesDoNotRetain) {
  PagePool pool(8);
  std::vector<kvlab::PageLease> owners;
  for (int i = 0; i < 8; ++i) {
    auto page = kvlab::PageLease::try_allocate(pool);
    ASSERT_TRUE(page);
    const auto borrowed = page->handle();
    const auto copied = borrowed;
    owners.push_back(std::move(*page));
    EXPECT_FALSE(*page);
    EXPECT_EQ(pool.inspect(*copied)->references, 1U);
  }
  owners.clear();
  EXPECT_EQ(pool.stats().free_pages, 8U);
  EXPECT_NO_THROW(pool.validate_invariants());
}

}  // namespace
}  // namespace kvlab
