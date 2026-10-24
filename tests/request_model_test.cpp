#include "kvlab/request_state.hpp"

#include <gtest/gtest.h>

#include <array>
#include <random>
#include <set>
#include <vector>

namespace kvlab {
namespace {

struct ModelRequest {
  std::size_t length = 0;
  std::size_t reserved = 0;
  std::vector<PageHandle> pages;
};

TEST(RequestModel, SeededReservationsCommitsSharingAndCancellationAgree) {
  constexpr std::size_t capacity = 13;
  constexpr std::size_t page_tokens = 4;
  for (std::uint32_t seed = 0; seed < 32; ++seed) {
    SCOPED_TRACE(::testing::Message() << "seed=" << seed);
    PagePool pool(capacity, page_tokens);
    std::array<RequestState, 4> requests{
        RequestState(pool), RequestState(pool), RequestState(pool), RequestState(pool)};
    std::array<ModelRequest, 4> model;
    std::array<std::uint64_t, capacity> latest_generation{};
    std::mt19937 random(seed);
    for (int step = 0; step < 2000; ++step) {
      const auto index = random() % requests.size();
      const auto operation = random() % 4;
      SCOPED_TRACE(::testing::Message() << "step=" << step << ", request=" << index
                                      << ", operation=" << operation);
      auto& actual = requests[index];
      auto& expected = model[index];
      std::set<PageId> live_ids;
      for (const auto& request : model) {
        for (const auto handle : request.pages) {
          live_ids.insert(handle.id);
        }
      }
      if (operation == 0) {
        const std::size_t target = random() % 49;
        // Independent formula, safe because model targets are bounded to 48.
        const auto pages = (target + 3) / 4;
        const bool fits = target <= expected.reserved ||
                          pages - expected.pages.size() <= capacity - live_ids.size();
        const auto status = actual.reserve_for_length(target);
        ASSERT_EQ(status, fits ? RequestStatus::Ok : RequestStatus::PoolExhausted);
        if (fits && target > expected.reserved) {
          for (std::size_t i = expected.pages.size(); i < pages; ++i) {
            const auto handle = actual.page_handle(i);
            ASSERT_TRUE(handle);
            ASSERT_LT(handle->id, capacity);
            ASSERT_TRUE(live_ids.insert(handle->id).second);
            EXPECT_GT(handle->generation, latest_generation[handle->id]);
            latest_generation[handle->id] = handle->generation;
            expected.pages.push_back(*handle);
          }
          expected.reserved = target;
        }
      } else if (operation == 1) {
        const std::size_t target = random() % 49;
        const bool valid = target >= expected.length && target <= expected.reserved;
        ASSERT_EQ(actual.commit_length(target),
                  valid ? RequestStatus::Ok : RequestStatus::InvalidLength);
        if (valid) {
          expected.length = target;
        }
      } else if (operation == 2) {
        const auto source = (index + 1 + random() % 3) % requests.size();
        const std::size_t full_pages = random() % 6;
        const auto status = actual.share_prefix_from(requests[source], full_pages);
        if (!expected.pages.empty()) {
          ASSERT_EQ(status, RequestStatus::NotEmpty);
        } else if (full_pages > model[source].length / 4) {
          ASSERT_EQ(status, RequestStatus::InvalidLength);
        } else {
          ASSERT_EQ(status, RequestStatus::Ok);
          expected.pages.assign(model[source].pages.begin(),
                                model[source].pages.begin() + static_cast<std::ptrdiff_t>(full_pages));
          expected.length = full_pages * 4;
          expected.reserved = expected.length;
        }
      } else {
        actual.reset();
        expected = ModelRequest{};
      }
      // Derive references and sealing from all logical owners, not from pool counters.
      std::array<std::uint32_t, capacity> references{};
      std::array<bool, capacity> sealed{};
      for (std::size_t r = 0; r < requests.size(); ++r) {
        EXPECT_EQ(requests[r].length(), model[r].length);
        EXPECT_EQ(requests[r].reserved_length(), model[r].reserved);
        ASSERT_EQ(requests[r].page_count(), model[r].pages.size());
        std::set<PageId> unique;
        for (std::size_t p = 0; p < model[r].pages.size(); ++p) {
          const auto wanted = model[r].pages[p];
          const auto got = requests[r].page_handle(p);
          ASSERT_TRUE(got);
          EXPECT_EQ(got->id, wanted.id);
          EXPECT_EQ(got->generation, wanted.generation);
          EXPECT_TRUE(unique.insert(wanted.id).second);
          ++references[wanted.id];
          sealed[wanted.id] = sealed[wanted.id] || p < model[r].length / 4;
        }
        EXPECT_FALSE(requests[r].token_location(model[r].length));
        if (model[r].length != 0) {
          const auto last = model[r].length - 1;
          const auto location = requests[r].token_location(last);
          ASSERT_TRUE(location);
          EXPECT_EQ(location->page.id, model[r].pages[last / 4].id);
          EXPECT_EQ(location->offset, last % 4);
        }
        if (step % 50 == 0) {
          EXPECT_NO_THROW(requests[r].validate_invariants());
        }
      }
      std::size_t occupied = 0;
      for (PageId id = 0; id < capacity; ++id) {
        const auto info = pool.inspect({id, latest_generation[id]});
        ASSERT_EQ(info.has_value(), references[id] != 0);
        if (info) {
          ++occupied;
          EXPECT_EQ(info->references, references[id]);
          EXPECT_EQ(info->sealed, sealed[id]);
        }
      }
      EXPECT_EQ(pool.stats().allocated_pages, occupied);
      EXPECT_EQ(pool.stats().free_pages, capacity - occupied);
      if (step % 50 == 0) {
        EXPECT_NO_THROW(pool.validate_invariants());
      }
    }
    for (auto& request : requests) {
      request.reset();
    }
    EXPECT_EQ(pool.stats().free_pages, capacity);
    EXPECT_NO_THROW(pool.validate_invariants());
  }
}

}  // namespace
}  // namespace kvlab
