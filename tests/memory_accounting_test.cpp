#include "kvlab/memory_accounting.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kvlab {
namespace {
void materialize(RequestState& request, std::size_t tokens) {
  ASSERT_EQ(request.reserve_for_length(tokens), RequestStatus::Ok);
  ASSERT_EQ(request.commit_length(tokens), RequestStatus::Ok);
}

TEST(MemoryAccounting, EmptyPoolAndExplicitKvGeometry) {
  PagePool pool(0, 4);
  const auto m = account_memory(pool, {}, {2, 3, 5, 2});
  EXPECT_EQ(m.bytes_per_token, 120u);
  EXPECT_EQ(m.pool_pages, 0u);
  EXPECT_EQ(m.live_capacity_tokens, 0u);
  EXPECT_EQ(m.modeled_pool_bytes, 0u);
}

TEST(MemoryAccounting, SeparatesCommittedReservedSlackAndFreeCapacity) {
  PagePool pool(4, 4);
  RequestState request(pool);
  ASSERT_EQ(request.reserve_for_length(9), RequestStatus::Ok);
  ASSERT_EQ(request.commit_length(5), RequestStatus::Ok);
  const auto m = account_memory(pool, {&request});
  EXPECT_EQ(m.live_pages, 3u);
  EXPECT_EQ(m.free_pages, 1u);
  EXPECT_EQ(m.logical_committed_tokens, 5u);
  EXPECT_EQ(m.logical_reserved_tokens, 9u);
  EXPECT_EQ(m.unique_committed_tokens, 5u);
  EXPECT_EQ(m.unique_reserved_tokens, 9u);
  EXPECT_EQ(m.reserved_uncommitted_tokens, 4u);
  EXPECT_EQ(m.tail_slack_tokens, 3u);
  EXPECT_EQ(m.shared_capacity_saved_tokens, 0u);
  EXPECT_EQ(m.live_capacity_tokens, 12u);
  EXPECT_EQ(m.modeled_pool_bytes, 16u * 2048);
  EXPECT_EQ(m.modeled_live_bytes + m.modeled_free_bytes, m.modeled_pool_bytes);
}

TEST(MemoryAccounting, CountsSharedPagesOnceAndIncludesPrefixOwner) {
  PagePool pool(8, 4);
  RequestState prefix(pool), a(pool), b(pool);
  materialize(prefix, 8);
  ASSERT_EQ(a.share_prefix_from(prefix, 2), RequestStatus::Ok);
  ASSERT_EQ(b.share_prefix_from(prefix, 2), RequestStatus::Ok);
  materialize(a, 9);
  materialize(b, 10);
  ASSERT_EQ(b.reserve_for_length(11), RequestStatus::Ok);
  const auto m = account_memory(pool, {&prefix, &a, &b}, {}, &prefix);
  EXPECT_EQ(m.cached_only_pages, 0u);
  EXPECT_EQ(m.logical_page_references, 8u);
  EXPECT_EQ(m.live_pages, 4u);
  EXPECT_EQ(m.logical_committed_tokens, 27u);
  EXPECT_EQ(m.unique_committed_tokens, 11u);
  EXPECT_EQ(m.unique_reserved_tokens, 12u);
  EXPECT_EQ(m.reserved_uncommitted_tokens, 1u);
  EXPECT_EQ(m.tail_slack_tokens, 4u);
  EXPECT_EQ(m.shared_capacity_saved_tokens, 16u);
  EXPECT_THROW(account_memory(pool, {&a, &b}), std::invalid_argument);
  a.reset();
  b.reset();
  EXPECT_EQ(account_memory(pool, {&prefix}, {}, &prefix).cached_only_pages, 2u);
  EXPECT_THROW(account_memory(pool, {&prefix}, {}, &a), std::invalid_argument);
  prefix.reset();
  EXPECT_EQ(account_memory(pool, {}).free_pages, 8u);
}

TEST(MemoryAccounting, RejectsUnrepresentedLeaseAndMissingRequest) {
  PagePool pool(2, 4);
  RequestState request(pool);
  materialize(request, 1);
  EXPECT_THROW(account_memory(pool, {}), std::invalid_argument);
  auto lease = PageLease::try_allocate(pool);
  ASSERT_TRUE(lease);
  EXPECT_THROW(account_memory(pool, {&request}), std::invalid_argument);
}

TEST(MemoryAccounting, RejectsNullDuplicateForeignAndMovedFromOwners) {
  PagePool pool(1, 4), foreign_pool(1, 4);
  RequestState a(pool), b(foreign_pool);
  materialize(a, 1);
  materialize(b, 1);  // Same numeric ID/generation must not disguise a foreign owner.
  EXPECT_THROW(account_memory(pool, {nullptr}), std::invalid_argument);
  EXPECT_THROW(account_memory(pool, {&a, &a}), std::invalid_argument);
  EXPECT_THROW(account_memory(pool, {&b}), std::invalid_argument);
  RequestState moved(std::move(a));
  EXPECT_THROW(account_memory(pool, {&a, &moved}), std::invalid_argument);
  EXPECT_EQ(account_memory(pool, {&moved}).live_pages, 1u);
}

TEST(MemoryAccounting, RejectsZeroDimensionsAndModeledByteOverflow) {
  PagePool pool(1, 4);
  EXPECT_THROW(account_memory(pool, {}, {0, 1, 1, 1}), std::invalid_argument);
  EXPECT_THROW(account_memory(pool, {}, {1, 0, 1, 1}), std::invalid_argument);
  EXPECT_THROW(account_memory(pool, {}, {1, 1, 0, 1}), std::invalid_argument);
  EXPECT_THROW(account_memory(pool, {}, {1, 1, 1, 0}), std::invalid_argument);
  EXPECT_THROW(account_memory(pool, {}, {UINT64_MAX, 1, 1, 1}), std::overflow_error);
  PagePool huge_page(1, std::numeric_limits<std::size_t>::max());
  EXPECT_THROW(account_memory(huge_page, {}, {1, 1, 1, 2}), std::overflow_error);
}

TEST(MemoryAccounting, ChecksLogicalSumEvenWhenPhysicalBytesFit) {
  if (sizeof(std::size_t) != sizeof(std::uint64_t)) GTEST_SKIP();
  const auto tokens = std::numeric_limits<std::size_t>::max() / 4;
  PagePool pool(1, tokens);
  RequestState prefix(pool), a(pool), b(pool), c(pool), d(pool);
  materialize(prefix, tokens);
  for (auto* request : {&a, &b, &c, &d}) {
    ASSERT_EQ(request->share_prefix_from(prefix, 1), RequestStatus::Ok);
  }
  EXPECT_THROW(account_memory(pool, {&prefix, &a, &b, &c, &d}, {1, 1, 1, 1}), std::overflow_error);
}
}  // namespace
}  // namespace kvlab
