#include "kvlab/workload.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

namespace kvlab {
namespace {
void require_ok(RequestStatus status) {
  if (status != RequestStatus::Ok) {
    throw std::logic_error("synthetic workload violated request contract");
  }
}
std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::overflow_error("metadata allocation counter overflow");
  }
  return a + b;
}
}  // namespace

const char* name(EventKind kind) noexcept {
  switch (kind) {
    case EventKind::Admit: return "admit";
    case EventKind::Append: return "append";
    case EventKind::Finish: return "finish";
    case EventKind::Cancel: return "cancel";
  }
  return "invalid";
}
const char* name(EventOutcome outcome) noexcept {
  switch (outcome) {
    case EventOutcome::Admitted: return "admitted";
    case EventOutcome::AdmissionRejected: return "admission_rejected";
    case EventOutcome::Appended: return "appended";
    case EventOutcome::GrowthRejected: return "growth_rejected";
    case EventOutcome::Finished: return "finished";
    case EventOutcome::Cancelled: return "cancelled";
    case EventOutcome::Skipped: return "skipped";
  }
  return "invalid";
}

std::vector<WorkloadEvent> generate_trace(std::uint32_t seed, std::size_t count,
                                        std::size_t slots, std::size_t max_tokens) {
  if (!slots || !max_tokens || slots > UINT32_MAX || max_tokens > UINT32_MAX) {
    throw std::invalid_argument("trace needs positive uint32-representable slots and max_tokens");
  }
  std::mt19937 random(seed);
  std::vector<WorkloadEvent> trace;
  trace.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto draw = random() % 100;
    const auto kind = draw < 25 ? EventKind::Admit : draw < 75 ? EventKind::Append
                                                   : draw < 87 ? EventKind::Finish : EventKind::Cancel;
    const auto slot = random() % slots;
    const auto token_draw = random();
    const auto tokens = kind == EventKind::Admit ? 1 + token_draw % max_tokens : 1 + token_draw % 16;
    const auto share = static_cast<unsigned>(random() % 100);
    trace.push_back({kind, slot, tokens, share});
  }
  return trace;
}

void* MetadataResource::do_allocate(std::size_t bytes, std::size_t alignment) {
  const auto calls = checked_add(stats_.allocation_calls, 1);
  const auto requested = checked_add(stats_.requested_bytes, bytes);
  const auto live = checked_add(stats_.live_bytes, bytes);
  auto* pointer = std::pmr::new_delete_resource()->allocate(bytes, alignment);
  stats_ = {calls, requested, live, std::max(stats_.peak_live_bytes, live)};
  return pointer;
}
void MetadataResource::do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) {
  std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
  stats_.live_bytes -= bytes;
}

SyntheticWorkload::SyntheticWorkload(WorkloadConfig config)
    : config_(config), pool_(config.pool_pages, config.tokens_per_page),
      prefix_(pool_, &metadata_) {
  if (!config.slots || !config.max_tokens || config.prefix_tokens > config.max_tokens ||
      config.share_percent > 100) {
    throw std::invalid_argument("invalid workload configuration");
  }
  // Cache only complete pages. Different P can change shareable prefix coverage.
  const auto prefix_length = config.share_percent == 0 ? 0
      : config.prefix_tokens - config.prefix_tokens % config.tokens_per_page;
  if (prefix_.reserve_for_length(prefix_length) == RequestStatus::PoolExhausted) {
    throw std::invalid_argument("pool cannot hold the configured prefix owner");
  }
  require_ok(prefix_.commit_length(prefix_length));
  requests_.reserve(config.slots);
  for (std::size_t i = 0; i < config.slots; ++i) {
    requests_.emplace_back(pool_, &metadata_);
  }
  active_.resize(config.slots, false);
}

EventOutcome SyntheticWorkload::apply(const WorkloadEvent& event) {
  if (drained_ || event.slot >= requests_.size() || !event.tokens || event.share_draw >= 100 ||
      (event.kind == EventKind::Admit && event.tokens > config_.max_tokens)) {
    throw std::invalid_argument("invalid workload event");
  }
  auto& request = requests_[event.slot];
  switch (event.kind) {
    case EventKind::Admit: {
      if (active_[event.slot]) return EventOutcome::Skipped;
      try {
        if (event.share_draw < config_.share_percent) {
          const auto pages = std::min(event.tokens, prefix_.length()) / config_.tokens_per_page;
          require_ok(request.share_prefix_from(prefix_, pages));
        }
        const auto status = request.reserve_for_length(event.tokens);
        if (status == RequestStatus::PoolExhausted) {
          request.reset();
          return EventOutcome::AdmissionRejected;
        }
        require_ok(status);
        require_ok(request.commit_length(event.tokens));
        active_[event.slot] = true;
        return EventOutcome::Admitted;
      } catch (...) {
        request.reset();  // Undo prefix retention if metadata allocation throws.
        throw;
      }
    }
    case EventKind::Append: {
      if (!active_[event.slot] || request.length() == config_.max_tokens) return EventOutcome::Skipped;
      const auto target = request.length() + std::min(event.tokens, config_.max_tokens - request.length());
      const auto status = request.reserve_for_length(target);
      if (status == RequestStatus::PoolExhausted) return EventOutcome::GrowthRejected;
      require_ok(status);
      require_ok(request.commit_length(target));
      return EventOutcome::Appended;
    }
    case EventKind::Finish:
    case EventKind::Cancel:
      if (!active_[event.slot]) return EventOutcome::Skipped;
      request.reset();
      active_[event.slot] = false;
      return event.kind == EventKind::Finish ? EventOutcome::Finished : EventOutcome::Cancelled;
  }
  throw std::invalid_argument("invalid event kind");
}

MemorySnapshot SyntheticWorkload::snapshot(KvGeometry geometry) const {
  std::vector<const RequestState*> owners{&prefix_};
  for (const auto& request : requests_) owners.push_back(&request);
  return account_memory(pool_, owners, geometry, &prefix_);
}

void SyntheticWorkload::drain() noexcept {
  for (auto& request : requests_) request.reset();
  std::fill(active_.begin(), active_.end(), false);
  prefix_.reset();
  drained_ = true;
}

}  // namespace kvlab
