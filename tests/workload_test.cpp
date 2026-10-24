#include "kvlab/workload.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace kvlab {
namespace {
TEST(Workload, FailedAdmissionRollsBackSharedPrefixAndCleanupRetainsCacheUntilDrain) {
  SyntheticWorkload workload({4, 4, 2, 12, 4, 100});
  EXPECT_EQ(workload.apply({EventKind::Admit, 0, 9, 0}), EventOutcome::Admitted);
  EXPECT_EQ(workload.apply({EventKind::Admit, 1, 9, 0}), EventOutcome::AdmissionRejected);
  const auto m = workload.snapshot();
  EXPECT_EQ(m.live_pages, 3u);
  EXPECT_EQ(m.logical_committed_tokens, 13u);
  EXPECT_EQ(m.logical_page_references, 4u);
  EXPECT_EQ(workload.apply({EventKind::Finish, 0, 1, 0}), EventOutcome::Finished);
  EXPECT_EQ(workload.snapshot().cached_only_pages, 1u);
  workload.drain();
  workload.drain();
  EXPECT_EQ(workload.snapshot().free_pages, 4u);
  EXPECT_THROW(workload.apply({EventKind::Admit, 0, 1, 0}), std::invalid_argument);
}

TEST(Workload, RejectedGrowthIsRetryableAfterCancellation) {
  SyntheticWorkload workload({2, 4, 2, 12, 0, 0});
  ASSERT_EQ(workload.apply({EventKind::Admit, 0, 4, 0}), EventOutcome::Admitted);
  ASSERT_EQ(workload.apply({EventKind::Admit, 1, 4, 0}), EventOutcome::Admitted);
  EXPECT_EQ(workload.apply({EventKind::Append, 0, 1, 0}), EventOutcome::GrowthRejected);
  EXPECT_EQ(workload.snapshot().logical_committed_tokens, 8u);
  EXPECT_EQ(workload.apply({EventKind::Cancel, 1, 1, 0}), EventOutcome::Cancelled);
  EXPECT_EQ(workload.apply({EventKind::Append, 0, 1, 0}), EventOutcome::Appended);
  EXPECT_EQ(workload.snapshot().logical_committed_tokens, 5u);
  workload.drain();
  EXPECT_EQ(workload.pool_stats().free_pages, 2u);
}

TEST(Workload, MetadataCounterCapturesGrowthButNotWithinPageAppend) {
  SyntheticWorkload workload({4, 4, 1, 16, 0, 0});
  EXPECT_EQ(workload.metadata_stats().allocation_calls, 0u);
  ASSERT_EQ(workload.apply({EventKind::Admit, 0, 1, 0}), EventOutcome::Admitted);
  const auto first = workload.metadata_stats();
  ASSERT_EQ(first.allocation_calls, 1u);
  EXPECT_EQ(workload.apply({EventKind::Append, 0, 3, 0}), EventOutcome::Appended);
  EXPECT_EQ(workload.metadata_stats().allocation_calls, first.allocation_calls);
  EXPECT_EQ(workload.apply({EventKind::Append, 0, 1, 0}), EventOutcome::Appended);
  const auto second = workload.metadata_stats();
  EXPECT_EQ(second.allocation_calls, 2u);
  EXPECT_GT(second.peak_live_bytes, second.live_bytes);  // Candidate + previous table coexist.
  EXPECT_EQ(second.requested_bytes, second.peak_live_bytes);
  workload.drain();
  EXPECT_EQ(workload.pool_stats().allocated_pages, 0u);
  EXPECT_EQ(workload.metadata_stats().live_bytes, second.live_bytes);  // reset keeps vector capacity.
}

TEST(Workload, TraceHasPortableGoldenPrefixAndReproducibleEvents) {
  const auto trace = generate_trace(5489, 100, 4, 16);
  ASSERT_EQ(trace.size(), 100u);
  EXPECT_EQ(trace[0].kind, EventKind::Admit);
  EXPECT_EQ(trace[0].slot, 2u);
  EXPECT_EQ(trace[0].tokens, 15u);
  EXPECT_EQ(trace[0].share_draw, 85u);
  const auto repeat = generate_trace(5489, 100, 4, 16);
  for (std::size_t i = 0; i < trace.size(); ++i) {
    EXPECT_EQ(trace[i].kind, repeat[i].kind);
    EXPECT_EQ(trace[i].slot, repeat[i].slot);
    EXPECT_EQ(trace[i].tokens, repeat[i].tokens);
    EXPECT_EQ(trace[i].share_draw, repeat[i].share_draw);
  }
  EXPECT_THROW(generate_trace(1, 1, 0, 1), std::invalid_argument);
  EXPECT_THROW(generate_trace(1, 1, 1, 0), std::invalid_argument);
}

TEST(Workload, ValidatesConfigurationAndEventsBeforeMutation) {
  EXPECT_THROW((SyntheticWorkload{WorkloadConfig{1, 4, 0, 4, 0, 0}}), std::invalid_argument);
  EXPECT_THROW((SyntheticWorkload{WorkloadConfig{1, 4, 1, 4, 8, 100}}), std::invalid_argument);
  EXPECT_THROW((SyntheticWorkload{WorkloadConfig{1, 4, 1, 8, 8, 100}}), std::invalid_argument);
  EXPECT_THROW((SyntheticWorkload{WorkloadConfig{1, 4, 1, 8, 0, 101}}), std::invalid_argument);
  SyntheticWorkload workload({1, 4, 1, 8, 0, 0});
  EXPECT_THROW(workload.apply({EventKind::Admit, 1, 1, 0}), std::invalid_argument);
  EXPECT_THROW(workload.apply({EventKind::Admit, 0, 9, 0}), std::invalid_argument);
  EXPECT_THROW(workload.apply({EventKind::Admit, 0, 1, 100}), std::invalid_argument);
  EXPECT_THROW(workload.apply({static_cast<EventKind>(99), 0, 1, 0}), std::invalid_argument);
  EXPECT_EQ(workload.snapshot().live_pages, 0u);
}

TEST(Workload, SeededUnsharedReplayMatchesIndependentLengthModel) {
  for (const std::size_t page_tokens : {1, 4, 16, 64}) {
    for (std::uint32_t seed = 0; seed < 4; ++seed) {
      SCOPED_TRACE(seed);
      const std::size_t capacity = 16, max_tokens = 129, slots = 8;
      SyntheticWorkload workload({capacity, page_tokens, slots, max_tokens, 0, 0});
      std::vector<std::size_t> lengths(slots);
      auto pages = [page_tokens](std::size_t n) { return n / page_tokens + (n % page_tokens != 0); };
      for (const auto& event : generate_trace(seed, 2000, slots, max_tokens)) {
        std::size_t allocated = 0;
        for (const auto length : lengths) allocated += pages(length);
        auto& length = lengths[event.slot];
        auto expected = EventOutcome::Skipped;
        if (event.kind == EventKind::Admit && length == 0) {
          if (pages(event.tokens) <= capacity - allocated) {
            length = event.tokens;
            expected = EventOutcome::Admitted;
          } else expected = EventOutcome::AdmissionRejected;
        } else if (event.kind == EventKind::Append && length && length < max_tokens) {
          const auto target = length + std::min(event.tokens, max_tokens - length);
          if (pages(target) - pages(length) <= capacity - allocated) {
            length = target;
            expected = EventOutcome::Appended;
          } else expected = EventOutcome::GrowthRejected;
        } else if ((event.kind == EventKind::Finish || event.kind == EventKind::Cancel) && length) {
          length = 0;
          expected = event.kind == EventKind::Finish ? EventOutcome::Finished : EventOutcome::Cancelled;
        }
        ASSERT_EQ(workload.apply(event), expected);
        const auto m = workload.snapshot();
        std::size_t expected_pages = 0;
        for (const auto n : lengths) expected_pages += pages(n);
        EXPECT_EQ(m.live_pages, expected_pages);
        EXPECT_EQ(m.unique_committed_tokens, std::accumulate(lengths.begin(), lengths.end(), std::size_t{0}));
        EXPECT_EQ(m.reserved_uncommitted_tokens, 0u);
      }
      workload.drain();
      EXPECT_EQ(workload.snapshot().free_pages, capacity);
    }
  }
}

TEST(Workload, SharedChurnReconcilesEveryOwnerAndRecoversCapacity) {
  for (std::uint32_t seed = 0; seed < 8; ++seed) {
    SyntheticWorkload workload({32, 4, 8, 129, 16, 90});
    for (const auto& event : generate_trace(seed, 1000, 8, 129)) {
      static_cast<void>(workload.apply(event));
      const auto m = workload.snapshot();
      EXPECT_EQ(m.free_pages + m.live_pages, m.pool_pages);
      EXPECT_EQ(m.unique_committed_tokens + m.reserved_uncommitted_tokens + m.tail_slack_tokens,
                m.live_capacity_tokens);
      EXPECT_GE(m.logical_committed_tokens, m.unique_committed_tokens);
    }
    workload.drain();
    EXPECT_EQ(workload.snapshot().live_pages, 0u);
  }
}
}  // namespace
}  // namespace kvlab
