#include "kvlab/request_state.hpp"
#include "support/failing_resource.hpp"
#include "support/page_pool_test_access.hpp"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace kvlab {
namespace {

static_assert(!std::is_copy_constructible_v<RequestState>);
static_assert(!std::is_copy_assignable_v<RequestState>);
static_assert(std::is_nothrow_move_constructible_v<RequestState>);
static_assert(!std::is_nothrow_move_assignable_v<RequestState>);

struct Snapshot {
  std::size_t length, reserved;
  std::vector<std::pair<PageId, std::uint64_t>> handles;
  bool operator==(const Snapshot& other) const {
    return length == other.length && reserved == other.reserved && handles == other.handles;
  }
};
Snapshot snapshot(const RequestState& request) {
  Snapshot result{request.length(), request.reserved_length(), {}};
  for (std::size_t i = 0; i < request.page_count(); ++i) {
    const auto handle = request.page_handle(i).value();
    result.handles.emplace_back(handle.id, handle.generation);
  }
  return result;
}

TEST(RequestState, EmptyRequestAndResourceValidation) {
  PagePool pool(0, 4);
  EXPECT_THROW((kvlab::RequestState{pool, nullptr}), std::invalid_argument);
  kvlab::RequestState request(pool);
  EXPECT_TRUE(request.attached());
  EXPECT_EQ(request.length(), 0U);
  EXPECT_EQ(request.page_count(), 0U);
  EXPECT_FALSE(request.page_handle(0));
  EXPECT_FALSE(request.token_location(0));
  EXPECT_EQ(request.reserve_for_length(0), RequestStatus::Ok);
  EXPECT_EQ(request.commit_length(0), RequestStatus::Ok);
  EXPECT_EQ(request.reserve_for_length(1), RequestStatus::PoolExhausted);
  request.reset();
  EXPECT_NO_THROW(request.validate_invariants());
}

TEST(RequestState, PageBoundariesAndCommittedAddressTranslation) {
  PagePool pool(4, 4);
  kvlab::RequestState request(pool);
  for (const auto length : std::array<std::size_t, 6>{0, 1, 3, 4, 5, 9}) {
    ASSERT_EQ(request.reserve_for_length(length), RequestStatus::Ok);
    EXPECT_EQ(request.page_count(), (length + 3) / 4);
    ASSERT_EQ(request.commit_length(length), RequestStatus::Ok);
    for (std::size_t position = 0; position < length; ++position) {
      const auto location = request.token_location(position);
      ASSERT_TRUE(location);
      EXPECT_EQ(location->page.id, request.page_handle(position / 4)->id);
      EXPECT_EQ(location->offset, position % 4);
    }
    EXPECT_FALSE(request.token_location(length));
    EXPECT_NO_THROW(request.validate_invariants());
  }
  request.reset();
  EXPECT_EQ(pool.stats().free_pages, 4U);
}

TEST(RequestState, ReservingAheadDoesNotCommitOrSealTokens) {
  PagePool pool(3, 4);
  kvlab::RequestState request(pool);
  ASSERT_EQ(request.reserve_for_length(9), RequestStatus::Ok);
  EXPECT_EQ(request.length(), 0U);
  EXPECT_EQ(request.reserved_length(), 9U);
  EXPECT_FALSE(request.token_location(0));
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(pool.inspect(*request.page_handle(i))->writable());
  }
  ASSERT_EQ(request.commit_length(5), RequestStatus::Ok);
  EXPECT_TRUE(pool.inspect(*request.page_handle(0))->sealed);
  EXPECT_TRUE(pool.inspect(*request.page_handle(1))->writable());
  EXPECT_TRUE(pool.inspect(*request.page_handle(2))->writable());
  EXPECT_EQ(request.reserved_length(), 9U);
  EXPECT_NO_THROW(request.validate_invariants());
}

TEST(RequestState, InvalidCommitLeavesMetadataUnchanged) {
  PagePool pool(2, 4);
  kvlab::RequestState request(pool);
  ASSERT_EQ(request.reserve_for_length(5), RequestStatus::Ok);
  ASSERT_EQ(request.commit_length(3), RequestStatus::Ok);
  const auto before = snapshot(request);
  EXPECT_EQ(request.commit_length(2), RequestStatus::InvalidLength);
  EXPECT_EQ(request.commit_length(6), RequestStatus::InvalidLength);
  EXPECT_TRUE(snapshot(request) == before);
  EXPECT_TRUE(pool.inspect(*request.page_handle(0))->writable());
  EXPECT_EQ(request.reserve_for_length(1), RequestStatus::Ok);
  EXPECT_TRUE(snapshot(request) == before);
}

TEST(RequestState, GrowthWithinReservedPageDoesNotAllocateHostMetadata) {
  PagePool pool(1, 4);
  test::FailingResource resource;
  kvlab::RequestState request(pool, &resource);
  ASSERT_EQ(request.reserve_for_length(1), RequestStatus::Ok);
  const auto allocations = resource.allocations;
  resource.fail_next = true;
  EXPECT_EQ(request.reserve_for_length(4), RequestStatus::Ok);
  EXPECT_EQ(resource.allocations, allocations);
  EXPECT_TRUE(resource.fail_next);
  EXPECT_EQ(request.commit_length(4), RequestStatus::Ok);
  resource.fail_next = false;
}

TEST(RequestState, CapacityFailurePreservesExistingTableAndCanRetry) {
  PagePool pool(3, 4);
  kvlab::RequestState request(pool);
  ASSERT_EQ(request.reserve_for_length(5), RequestStatus::Ok);
  ASSERT_EQ(request.commit_length(5), RequestStatus::Ok);
  auto blocker = PageLease::try_allocate(pool);
  ASSERT_TRUE(blocker);
  const auto before = snapshot(request);
  EXPECT_EQ(request.reserve_for_length(9), RequestStatus::PoolExhausted);
  EXPECT_TRUE(snapshot(request) == before);
  EXPECT_EQ(pool.stats().free_pages, 0U);
  blocker->reset();
  EXPECT_EQ(request.reserve_for_length(9), RequestStatus::Ok);
  EXPECT_EQ(request.commit_length(9), RequestStatus::Ok);
  EXPECT_NO_THROW(request.validate_invariants());
}

TEST(RequestState, HostAllocationFailureLeavesRequestAndPoolUnchanged) {
  PagePool pool(3, 4);
  test::FailingResource resource;
  {
    kvlab::RequestState request(pool, &resource);
    ASSERT_EQ(request.reserve_for_length(1), RequestStatus::Ok);
    ASSERT_EQ(request.commit_length(1), RequestStatus::Ok);
    const auto before = snapshot(request);
    const auto bytes = resource.outstanding_bytes;
    resource.fail_next = true;
    EXPECT_THROW(static_cast<void>(request.reserve_for_length(9)), std::bad_alloc);
    EXPECT_TRUE(snapshot(request) == before);
    EXPECT_EQ(resource.outstanding_bytes, bytes);
    EXPECT_EQ(pool.stats().free_pages, 2U);
    EXPECT_NO_THROW(request.validate_invariants());
  }
  EXPECT_EQ(resource.outstanding_bytes, 0U);
  EXPECT_EQ(pool.stats().free_pages, 3U);
}

TEST(RequestState, PartialAcquisitionExceptionRollsBackLeasesButNotGenerations) {
  PagePool pool(4, 4);
  kvlab::RequestState request(pool);
  ASSERT_EQ(request.reserve_for_length(1), RequestStatus::Ok);
  ASSERT_EQ(request.commit_length(1), RequestStatus::Ok);
  const auto before = snapshot(request);
  const auto first = detail::PagePoolTestAccess::next_free_id(pool, 0);
  const auto failing = detail::PagePoolTestAccess::next_free_id(pool, 1);
  const auto original_generation = detail::PagePoolTestAccess::generation(pool, first);
  detail::PagePoolTestAccess::set_free_generation(pool, failing,
                                                 std::numeric_limits<std::uint64_t>::max());
  EXPECT_THROW(static_cast<void>(request.reserve_for_length(9)), std::overflow_error);
  EXPECT_TRUE(snapshot(request) == before);
  EXPECT_EQ(pool.stats().free_pages, 3U);
  EXPECT_EQ(detail::PagePoolTestAccess::generation(pool, first), original_generation + 1);
  EXPECT_NO_THROW(request.validate_invariants());
  EXPECT_NO_THROW(pool.validate_invariants());
  detail::PagePoolTestAccess::set_free_generation(pool, failing, 0);
  EXPECT_EQ(request.reserve_for_length(9), RequestStatus::Ok);
}

TEST(RequestState, PrefixSharingKeepsPartialTailsIndependent) {
  PagePool pool(4, 4);
  kvlab::RequestState source(pool), child(pool);
  ASSERT_EQ(source.reserve_for_length(5), RequestStatus::Ok);
  ASSERT_EQ(source.commit_length(5), RequestStatus::Ok);
  EXPECT_EQ(child.share_prefix_from(source, 2), RequestStatus::InvalidLength);
  ASSERT_EQ(child.share_prefix_from(source, 1), RequestStatus::Ok);
  EXPECT_EQ(child.length(), 4U);
  EXPECT_EQ(child.page_handle(0)->id, source.page_handle(0)->id);
  EXPECT_EQ(pool.inspect(*source.page_handle(0))->references, 2U);
  ASSERT_EQ(child.reserve_for_length(5), RequestStatus::Ok);
  ASSERT_EQ(child.commit_length(5), RequestStatus::Ok);
  EXPECT_NE(child.page_handle(1)->id, source.page_handle(1)->id);
  EXPECT_EQ(pool.stats().allocated_pages, 3U);
  source.reset();
  EXPECT_EQ(pool.inspect(*child.page_handle(0))->references, 1U);
  EXPECT_TRUE(pool.inspect(*child.page_handle(0))->sealed);
  EXPECT_EQ(child.reserve_for_length(8), RequestStatus::Ok);
  EXPECT_EQ(child.commit_length(8), RequestStatus::Ok);
  child.reset();
  EXPECT_EQ(pool.stats().free_pages, 4U);
}

TEST(RequestState, PrefixValidationDoesNotAcquireReferences) {
  PagePool pool(3, 4), other_pool(1, 4);
  kvlab::RequestState source(pool), destination(pool), other(other_pool);
  ASSERT_EQ(source.reserve_for_length(8), RequestStatus::Ok);
  ASSERT_EQ(source.commit_length(4), RequestStatus::Ok);
  EXPECT_EQ(destination.share_prefix_from(source, 2), RequestStatus::InvalidLength);
  EXPECT_EQ(other.share_prefix_from(source, 1), RequestStatus::PoolMismatch);
  EXPECT_EQ(destination.share_prefix_from(source, 0), RequestStatus::Ok);
  ASSERT_EQ(destination.reserve_for_length(1), RequestStatus::Ok);
  EXPECT_EQ(destination.share_prefix_from(source, 1), RequestStatus::NotEmpty);
  EXPECT_EQ(source.share_prefix_from(source, 1), RequestStatus::NotEmpty);
  EXPECT_EQ(pool.inspect(*source.page_handle(0))->references, 1U);
}

TEST(RequestState, PrefixHostAllocationFailureDoesNotRetain) {
  PagePool pool(2, 4);
  test::FailingResource resource;
  kvlab::RequestState source(pool), destination(pool, &resource);
  ASSERT_EQ(source.reserve_for_length(8), RequestStatus::Ok);
  ASSERT_EQ(source.commit_length(8), RequestStatus::Ok);
  const auto before = snapshot(source);
  resource.fail_next = true;
  EXPECT_THROW(static_cast<void>(destination.share_prefix_from(source, 2)), std::bad_alloc);
  EXPECT_TRUE(snapshot(source) == before);
  EXPECT_EQ(destination.page_count(), 0U);
  EXPECT_EQ(resource.outstanding_bytes, 0U);
  EXPECT_EQ(pool.inspect(*source.page_handle(0))->references, 1U);
  EXPECT_EQ(pool.inspect(*source.page_handle(1))->references, 1U);
}

TEST(RequestState, LaterRetainFailureRollsBackEarlierPrefixShares) {
  PagePool pool(2, 4);
  kvlab::RequestState source(pool), destination(pool);
  ASSERT_EQ(source.reserve_for_length(8), RequestStatus::Ok);
  ASSERT_EQ(source.commit_length(8), RequestStatus::Ok);
  const auto saturated = *source.page_handle(1);
  const auto before = snapshot(source);
  detail::PagePoolTestAccess::set_references(pool, saturated,
                                            std::numeric_limits<std::uint32_t>::max());
  EXPECT_EQ(destination.share_prefix_from(source, 2), RequestStatus::NotShareable);
  EXPECT_TRUE(snapshot(source) == before);
  EXPECT_EQ(destination.length(), 0U);
  EXPECT_EQ(destination.page_count(), 0U);
  EXPECT_EQ(pool.inspect(*source.page_handle(0))->references, 1U);
  EXPECT_EQ(pool.inspect(saturated)->references, std::numeric_limits<std::uint32_t>::max());
  detail::PagePoolTestAccess::set_references(pool, saturated, 1);
  EXPECT_EQ(destination.share_prefix_from(source, 2), RequestStatus::Ok);
}

TEST(RequestState, CommitPreflightRejectsExternalSealingBeforePartialMutation) {
  PagePool pool(2, 4);
  kvlab::RequestState request(pool);
  ASSERT_EQ(request.reserve_for_length(8), RequestStatus::Ok);
  // Deliberate caller-contract violation: request metadata is normally the only writer.
  ASSERT_EQ(pool.seal_full(*request.page_handle(1), 4), SealResult::Sealed);
  EXPECT_EQ(request.commit_length(8), RequestStatus::NotWritable);
  EXPECT_EQ(request.length(), 0U);
  EXPECT_TRUE(pool.inspect(*request.page_handle(0))->writable());
  request.reset();
  EXPECT_EQ(pool.stats().free_pages, 2U);
}

TEST(RequestState, ExtremeLengthsDoNotOverflowPageArithmetic) {
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  PagePool small(1, 4);
  kvlab::RequestState rejected(small);
  EXPECT_EQ(rejected.reserve_for_length(maximum), RequestStatus::PoolExhausted);
  EXPECT_EQ(rejected.page_count(), 0U);
  // Metadata arithmetic only: this does not allocate SIZE_MAX tokens of payload.
  PagePool enormous_page(1, maximum);
  kvlab::RequestState source(enormous_page), shared(enormous_page);
  ASSERT_EQ(source.reserve_for_length(maximum), RequestStatus::Ok);
  ASSERT_EQ(source.commit_length(maximum), RequestStatus::Ok);
  ASSERT_TRUE(source.token_location(maximum - 1));
  EXPECT_EQ(source.token_location(maximum - 1)->offset, maximum - 1);
  EXPECT_FALSE(source.token_location(maximum));
  EXPECT_EQ(shared.share_prefix_from(source, 1), RequestStatus::Ok);
  EXPECT_EQ(shared.length(), maximum);
  EXPECT_NO_THROW(source.validate_invariants());
  EXPECT_NO_THROW(shared.validate_invariants());
}

TEST(RequestState, MoveConstructionTransfersWithoutRetainingAndDetachesSource) {
  PagePool pool(2, 4);
  kvlab::RequestState source(pool);
  ASSERT_EQ(source.reserve_for_length(5), RequestStatus::Ok);
  ASSERT_EQ(source.commit_length(4), RequestStatus::Ok);
  const auto before = snapshot(source);
  kvlab::RequestState moved(std::move(source));
  EXPECT_TRUE(snapshot(moved) == before);
  EXPECT_FALSE(source.attached());
  EXPECT_EQ(source.reserve_for_length(1), RequestStatus::MovedFrom);
  EXPECT_EQ(source.commit_length(0), RequestStatus::MovedFrom);
  EXPECT_EQ(source.share_prefix_from(moved, 0), RequestStatus::MovedFrom);
  EXPECT_EQ(moved.share_prefix_from(source, 0), RequestStatus::MovedFrom);
  source.reset();
  EXPECT_NO_THROW(source.validate_invariants());
  EXPECT_EQ(pool.inspect(*moved.page_handle(0))->references, 1U);
}

TEST(RequestState, MoveAssignmentFailurePreservesBothRequestsAndResources) {
  PagePool first(2, 4), second(2, 4);
  test::FailingResource first_resource, second_resource;
  {
    kvlab::RequestState destination(first, &first_resource), source(second, &second_resource);
    ASSERT_EQ(destination.reserve_for_length(1), RequestStatus::Ok);
    ASSERT_EQ(source.reserve_for_length(5), RequestStatus::Ok);
    ASSERT_EQ(source.commit_length(5), RequestStatus::Ok);
    const auto before_dest = snapshot(destination), before_source = snapshot(source);
    first_resource.fail_next = true;
    EXPECT_THROW(destination = std::move(source), std::bad_alloc);
    EXPECT_TRUE(snapshot(destination) == before_dest);
    EXPECT_TRUE(snapshot(source) == before_source);
    EXPECT_EQ(first.stats().free_pages, 1U);
    EXPECT_EQ(second.stats().free_pages, 0U);
    destination = std::move(source);
    EXPECT_TRUE(snapshot(destination) == before_source);
    EXPECT_FALSE(source.attached());
    EXPECT_EQ(first.stats().free_pages, 2U);
    EXPECT_EQ(second.stats().free_pages, 0U);
    EXPECT_NO_THROW(destination.validate_invariants());
  }
  EXPECT_EQ(first_resource.outstanding_bytes, 0U);
  EXPECT_EQ(second_resource.outstanding_bytes, 0U);
  EXPECT_EQ(second.stats().free_pages, 2U);
}

TEST(RequestState, SharedMoveAssignmentAndSelfMoveBalanceOwners) {
  PagePool pool(1, 4);
  kvlab::RequestState first(pool), second(pool);
  ASSERT_EQ(first.reserve_for_length(4), RequestStatus::Ok);
  ASSERT_EQ(first.commit_length(4), RequestStatus::Ok);
  ASSERT_EQ(second.share_prefix_from(first, 1), RequestStatus::Ok);
  const auto handle = *first.page_handle(0);
  first = std::move(second);
  EXPECT_EQ(pool.inspect(handle)->references, 1U);
  auto* alias = &first;
  first = std::move(*alias);
  EXPECT_EQ(pool.inspect(handle)->references, 1U);
  first.reset();
  EXPECT_EQ(pool.stats().free_pages, 1U);
}

TEST(RequestState, CancellationAndExceptionUnwindingReturnAllOwnedPages) {
  PagePool pool(3, 4);
  kvlab::RequestState source(pool);
  ASSERT_EQ(source.reserve_for_length(4), RequestStatus::Ok);
  ASSERT_EQ(source.commit_length(4), RequestStatus::Ok);
  const auto fail = [&] {
    kvlab::RequestState child(pool);
    if (child.share_prefix_from(source, 1) != RequestStatus::Ok ||
        child.reserve_for_length(9) != RequestStatus::Ok) {
      throw std::logic_error("unexpected setup failure");
    }
    throw std::runtime_error("synthetic cancellation");
  };
  EXPECT_THROW(fail(), std::runtime_error);
  EXPECT_EQ(pool.stats().allocated_pages, 1U);
  EXPECT_EQ(pool.inspect(*source.page_handle(0))->references, 1U);
  source.reset();
  source.reset();
  EXPECT_EQ(pool.stats().free_pages, 3U);
  EXPECT_EQ(source.reserve_for_length(1), RequestStatus::Ok);
}

}  // namespace
}  // namespace kvlab
