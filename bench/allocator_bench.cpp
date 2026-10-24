#include "kvlab/workload.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {
using Clock = std::chrono::steady_clock;
using Ns = std::chrono::nanoseconds;
using kvlab::EventKind;
using kvlab::EventOutcome;
struct Sample { EventOutcome outcome; std::int64_t ns; std::size_t metadata_allocations; };
struct MemorySample { std::size_t step; const char* phase; kvlab::MemorySnapshot value; };

std::size_t number(const std::string& value) {
  std::size_t parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
    throw std::invalid_argument("expected a nonnegative decimal integer: " + value);
  }
  return parsed;
}

std::ofstream output_file(const std::filesystem::path& path) {
  std::ofstream output;
  output.exceptions(std::ios::badbit | std::ios::failbit);
  output.open(path);
  return output;
}

std::int64_t percentile(const std::vector<std::int64_t>& sorted, std::size_t percent) {
  return sorted.at((sorted.size() * percent + 99) / 100 - 1);  // nearest rank
}

int run(int argc, char** argv) {
  std::map<std::string, std::string> options{
      {"--events", "20000"}, {"--seed", "1"}, {"--pool-tokens", "4096"},
      {"--page-tokens", "16"}, {"--slots", "32"}, {"--max-tokens", "1024"},
      {"--prefix-tokens", "128"}, {"--share-percent", "50"},
      {"--sample-every", "128"}, {"--output", ""}};
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "allocator_bench --output NEW_DIRECTORY [--events N --seed N --pool-tokens N "
                 "--page-tokens N --slots N --max-tokens N --prefix-tokens N "
                 "--share-percent N --sample-every N]\n";
    return 0;
  }
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc || !options.count(argv[i])) throw std::invalid_argument("unknown or incomplete option");
    options.at(argv[i]) = argv[i + 1];
  }
  const auto events = number(options.at("--events"));
  const auto seed = number(options.at("--seed"));
  const auto pool_tokens = number(options.at("--pool-tokens"));
  const auto page_tokens = number(options.at("--page-tokens"));
  const auto slots = number(options.at("--slots"));
  const auto max_tokens = number(options.at("--max-tokens"));
  const auto prefix_tokens = number(options.at("--prefix-tokens"));
  const auto share = number(options.at("--share-percent"));
  const auto every = number(options.at("--sample-every"));
  const std::filesystem::path directory = options.at("--output");
  // Bounded local harness, not a claim about the allocator's maximum dimensions.
  if (!events || events > 5000000 || seed > UINT32_MAX || !page_tokens ||
      !pool_tokens || pool_tokens > 16777216 || pool_tokens % page_tokens ||
      !slots || slots > 100000 || !max_tokens || max_tokens > 16777216 ||
      prefix_tokens > max_tokens || share > 100 || !every || directory.empty()) {
    throw std::invalid_argument("invalid or oversized benchmark configuration (see docs/BENCHMARKS.md)");
  }
  if (std::filesystem::exists(directory)) throw std::invalid_argument("output directory already exists");
  const kvlab::WorkloadConfig config{pool_tokens / page_tokens, page_tokens, slots,
                                      max_tokens, prefix_tokens, static_cast<unsigned>(share)};
  const auto trace = kvlab::generate_trace(static_cast<std::uint32_t>(seed), events, slots, max_tokens);
  const auto warmup_count = std::min<std::size_t>(events, 2048);
  {
    kvlab::SyntheticWorkload warmup(config);
    for (std::size_t i = 0; i < warmup_count; ++i) static_cast<void>(warmup.apply(trace[i]));
    static_cast<void>(warmup.snapshot());
    warmup.drain();
  }
  std::vector<std::int64_t> timer_control(10000);
  for (auto& ns : timer_control) {
    const auto start = Clock::now();
    const auto end = Clock::now();
    ns = std::chrono::duration_cast<Ns>(end - start).count();
  }
  std::sort(timer_control.begin(), timer_control.end());

  kvlab::SyntheticWorkload measured(config);
  static_cast<void>(measured.snapshot());
  std::vector<Sample> samples;
  samples.reserve(events);
  const auto loop_start = Clock::now();
  for (const auto& event : trace) {
    const auto calls_before = measured.metadata_stats().allocation_calls;
    const auto start = Clock::now();
    const auto outcome = measured.apply(event);
    const auto end = Clock::now();
    samples.push_back({outcome, std::chrono::duration_cast<Ns>(end - start).count(),
                       measured.metadata_stats().allocation_calls - calls_before});
  }
  const auto loop_ns = std::chrono::duration_cast<Ns>(Clock::now() - loop_start).count();
  const auto before_drain = measured.snapshot();
  const auto metadata = measured.metadata_stats();
  measured.drain();
  const auto after_drain = measured.snapshot();
  if (after_drain.live_pages != 0) throw std::logic_error("timed replay leaked pages");

  // Diagnostics have their own replay; page-table scans do not perturb each
  // individual timed operation. Assert the replay makes identical decisions.
  kvlab::SyntheticWorkload diagnostic(config);
  std::vector<MemorySample> memory{{0, "initial", diagnostic.snapshot()}};
  for (std::size_t i = 0; i < trace.size(); ++i) {
    if (diagnostic.apply(trace[i]) != samples[i].outcome) {
      throw std::logic_error("timed and diagnostic replay outcomes differ");
    }
    if ((i + 1) % every == 0 || i + 1 == events) {
      memory.push_back({i + 1, "sample", diagnostic.snapshot()});
    }
  }
  diagnostic.drain();
  memory.push_back({events, "drained", diagnostic.snapshot()});

  // Two clock reads amortize clock granularity over a whole trace. This is a
  // mean across all offered events (including skips/rejections), not event p99.
  kvlab::SyntheticWorkload aggregate(config);
  std::uint64_t checksum = 0, expected_checksum = 0;
  for (const auto& sample : samples) expected_checksum += static_cast<unsigned>(sample.outcome);
  const auto aggregate_start = Clock::now();
  for (const auto& event : trace) checksum += static_cast<unsigned>(aggregate.apply(event));
  const auto aggregate_ns = std::chrono::duration_cast<Ns>(Clock::now() - aggregate_start).count();
  if (checksum != expected_checksum || aggregate.snapshot().logical_committed_tokens != before_drain.logical_committed_tokens) {
    throw std::logic_error("aggregate replay disagrees with diagnostic replay");
  }
  aggregate.drain();
  if (aggregate.snapshot().live_pages != 0) throw std::logic_error("aggregate replay leaked pages");

  if (!std::filesystem::create_directories(directory)) throw std::runtime_error("cannot create new output directory");
  auto raw = output_file(directory / "events.csv");
  raw << "index,kind,slot,tokens,share_draw,outcome,elapsed_ns,metadata_allocations\n";
  using Key = std::tuple<EventKind, EventOutcome, bool>;
  std::map<Key, std::vector<std::int64_t>> distributions;
  std::array<std::size_t, 7> outcomes{};
  for (std::size_t i = 0; i < events; ++i) {
    const auto& event = trace[i];
    const auto& sample = samples[i];
    raw << i << ',' << kvlab::name(event.kind) << ',' << event.slot << ',' << event.tokens
        << ',' << event.share_draw << ',' << kvlab::name(sample.outcome) << ',' << sample.ns
        << ',' << sample.metadata_allocations << '\n';
    distributions[{event.kind, sample.outcome, sample.metadata_allocations != 0}].push_back(sample.ns);
    ++outcomes.at(static_cast<std::size_t>(sample.outcome));
  }
  raw.close();
  auto control = output_file(directory / "timer-control.csv");
  control << "elapsed_ns,count\n";
  std::map<std::int64_t, std::size_t> control_histogram;
  for (const auto ns : timer_control) ++control_histogram[ns];
  for (const auto& [ns, count] : control_histogram) control << ns << ',' << count << '\n';
  control.close();
  auto occupancy = output_file(directory / "memory.csv");
  occupancy << "step,phase,pool_pages,free_pages,live_pages,cached_only_pages,logical_page_references,logical_committed_tokens,"
               "logical_reserved_tokens,unique_committed_tokens,unique_reserved_tokens,live_capacity_tokens,"
               "reserved_uncommitted_tokens,tail_slack_tokens,shared_capacity_saved_tokens,bytes_per_token,"
               "modeled_pool_bytes,modeled_live_bytes,modeled_free_bytes\n";
  for (const auto& row : memory) {
    const auto& m = row.value;
    occupancy << row.step << ',' << row.phase << ',' << m.pool_pages << ',' << m.free_pages << ','
              << m.live_pages << ',' << m.cached_only_pages << ',' << m.logical_page_references << ',' << m.logical_committed_tokens << ','
              << m.logical_reserved_tokens << ',' << m.unique_committed_tokens << ',' << m.unique_reserved_tokens << ','
              << m.live_capacity_tokens << ',' << m.reserved_uncommitted_tokens << ',' << m.tail_slack_tokens << ','
              << m.shared_capacity_saved_tokens << ',' << m.bytes_per_token << ',' << m.modeled_pool_bytes << ','
              << m.modeled_live_bytes << ',' << m.modeled_free_bytes << '\n';
  }
  occupancy.close();

  auto summary = output_file(directory / "summary.json");
  summary << std::setprecision(12) << "{\n  \"schema_version\": 2,\n  \"measurement\": \"cpu_metadata_event\",\n"
          << "  \"clock\": \"steady_clock\",\n  \"build_type\": \"" << KVLAB_BUILD_TYPE << "\",\n"
          << "  \"config\": {\"events\": " << events << ", \"seed\": " << seed
          << ", \"pool_tokens\": " << pool_tokens << ", \"page_tokens\": " << page_tokens
          << ", \"slots\": " << slots << ", \"max_tokens\": " << max_tokens
          << ", \"prefix_tokens\": " << prefix_tokens << ", \"share_percent\": " << share
          << ", \"sample_every\": " << every << "},\n"
          << "  \"kv_model\": {\"layers\": 1, \"kv_heads\": 8, \"head_dimension\": 64, \"element_bytes\": 2},\n"
          << "  \"warmup_events\": " << warmup_count << ",\n  \"timer_control_p50_ns\": "
          << percentile(timer_control, 50) << ",\n  \"loop_elapsed_ns\": " << loop_ns
          << ",\n  \"timer_control_zero_samples\": " << control_histogram[0]
          << ",\n  \"aggregate_replay_ns\": " << aggregate_ns
          << ",\n  \"aggregate_ns_per_offered_event\": " << static_cast<double>(aggregate_ns) / events
          << ",\n  \"replay_events_per_second\": " << (loop_ns ? 1e9 * events / loop_ns : 0)
          << ",\n  \"live_pages_before_drain\": " << before_drain.live_pages
          << ",\n  \"free_pages_after_drain\": " << after_drain.free_pages << ",\n"
          << "  \"request_metadata\": {\"allocation_calls\": " << metadata.allocation_calls
          << ", \"requested_bytes\": " << metadata.requested_bytes
          << ", \"live_bytes_before_drain\": " << metadata.live_bytes
          << ", \"live_bytes_after_drain\": " << measured.metadata_stats().live_bytes
          << ", \"peak_live_bytes\": " << metadata.peak_live_bytes << "},\n  \"outcomes\": {";
  for (std::size_t i = 0; i < outcomes.size(); ++i) {
    summary << (i ? ", " : "") << '"' << kvlab::name(static_cast<EventOutcome>(i)) << "\": " << outcomes[i];
  }
  summary << "},\n  \"latency_by_operation_and_outcome\": [\n";
  bool first = true;
  for (auto& [key, values] : distributions) {
    std::sort(values.begin(), values.end());
    summary << (first ? "" : ",\n") << "    {\"operation\": \"" << kvlab::name(std::get<0>(key))
            << "\", \"outcome\": \"" << kvlab::name(std::get<1>(key))
            << "\", \"metadata_allocated\": " << (std::get<2>(key) ? "true" : "false")
            << ", \"count\": " << values.size()
            << ", \"zero_duration_samples\": " << std::count(values.begin(), values.end(), 0)
            << ", \"p50_ns\": " << percentile(values, 50) << ", \"p95_ns\": " << percentile(values, 95)
            << ", \"p99_ns\": ";
    if (values.size() >= 10000) summary << percentile(values, 99); else summary << "null";
    summary << '}';
    first = false;
  }
  summary << "\n  ]\n}\n";
  summary.close();
  std::cout << "Saved " << events << " CPU metadata events to " << directory << '\n';
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try { return run(argc, argv); }
  catch (const std::exception& error) {
    std::cerr << "allocator_bench: " << error.what() << '\n';
    return 1;
  }
}
