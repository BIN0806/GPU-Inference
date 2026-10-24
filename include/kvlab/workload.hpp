#pragma once

#include "kvlab/memory_accounting.hpp"

#include <memory_resource>
#include <vector>

namespace kvlab {

enum class EventKind { Admit, Append, Finish, Cancel };
enum class EventOutcome { Admitted, AdmissionRejected, Appended, GrowthRejected,
                          Finished, Cancelled, Skipped };
const char* name(EventKind kind) noexcept;
const char* name(EventOutcome outcome) noexcept;

struct WorkloadConfig {
  std::size_t pool_pages = 256, tokens_per_page = 16, slots = 32;
  std::size_t max_tokens = 1024, prefix_tokens = 128;
  unsigned share_percent = 50;
};

struct WorkloadEvent {
  EventKind kind;
  std::size_t slot, tokens;
  unsigned share_draw;  // 0..99, interpreted at replay time; same trace across policies.
};

// mt19937 raw outputs modulo the stated range: portable deterministic trace,
// with small modulo bias; not a distribution model of production traffic.
std::vector<WorkloadEvent> generate_trace(std::uint32_t seed, std::size_t count,
                                        std::size_t slots, std::size_t max_tokens);

struct MetadataStats {
  std::size_t allocation_calls = 0, requested_bytes = 0;
  std::size_t live_bytes = 0, peak_live_bytes = 0;
};

// Counts only RequestState PMR allocations, not all process heap allocations.
class MetadataResource final : public std::pmr::memory_resource {
 public:
  MetadataStats stats() const noexcept { return stats_; }

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override;
  void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override;
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
  MetadataStats stats_;
};

// Single-thread CPU metadata simulation. Prefix owner is explicitly included in
// snapshots and consumes pool space until drain(). No real KV writes or scheduler.
class SyntheticWorkload {
 public:
  explicit SyntheticWorkload(WorkloadConfig config);
  EventOutcome apply(const WorkloadEvent& event);
  MemorySnapshot snapshot(KvGeometry geometry = {}) const;
  MetadataStats metadata_stats() const noexcept { return metadata_.stats(); }
  void drain() noexcept;  // Terminal, idempotent cleanup; apply() then rejects events.
  PoolStats pool_stats() const noexcept { return pool_.stats(); }

 private:
  WorkloadConfig config_;
  PagePool pool_;
  MetadataResource metadata_;  // Outlives prefix_ and requests_.
  RequestState prefix_;
  std::vector<RequestState> requests_;
  std::vector<bool> active_;
  bool drained_ = false;
};

}  // namespace kvlab
