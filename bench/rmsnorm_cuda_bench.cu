#include "rmsnorm_cuda_bench.hpp"
#include "rmsnorm_launch.hpp"
#include "cuda_check_utils.cuh"

#include <cuda_fp16.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
using kvlab::detail::Variant;
using kvlab::test::check;
using kvlab::test::DeviceBuffer;
using kvlab::test::Stream;

struct Event {
  cudaEvent_t event{};
  Event() { check(cudaEventCreate(&event)); }
  ~Event() { cudaEventDestroy(event); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

void time_shape(int rows, int width) {
  constexpr int warmup = 24, pairs = 12, group_calls = 64;
  const std::size_t count = static_cast<std::size_t>(rows) * width;
  DeviceBuffer<__half> x(count), residual(count), weight(width), z(count), y(count);
  std::vector<__half> host_x(count), host_r(count), host_w(width);
  for (std::size_t i = 0; i < count; ++i) {
    host_x[i] = __float2half_rn((static_cast<int>(i % 53) - 26) / 13.0F);
    host_r[i] = __float2half_rn((static_cast<int>(i % 31) - 15) / 7.0F);
  }
  for (int i = 0; i < width; ++i)
    host_w[i] = __float2half_rn((i % 17 - 8) / 4.0F);

  Stream owner;
  const auto stream = owner.stream;
  check(cudaMemcpyAsync(x.data, host_x.data(), x.bytes, cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(residual.data, host_r.data(), residual.bytes, cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(weight.data, host_w.data(), weight.bytes, cudaMemcpyHostToDevice, stream));
  const kvlab::RmsNormArgs args{rows, width, kvlab::RmsNormType::Float16,
      {x.data, x.bytes}, {residual.data, residual.bytes}, {weight.data, weight.bytes},
      {z.data, z.bytes}, {y.data, y.bytes}};
  Event start, stop;
  for (int n = 0; n < warmup; ++n) {
    check(kvlab::residual_rmsnorm_unfused(args, stream));
    check(kvlab::residual_rmsnorm_fused(args, stream));
  }
  check(cudaStreamSynchronize(stream));

  for (int pair = 0; pair < pairs; ++pair) {
    for (int position = 0; position < 2; ++position) {
      const bool fused = (pair + position) % 2 != 0;
      const auto variant = fused ? Variant::Fused : Variant::Unfused;
      const auto checked = fused ? kvlab::residual_rmsnorm_fused : kvlab::residual_rmsnorm_unfused;
      check(cudaEventRecord(start.event, stream));
      for (int n = 0; n < group_calls; ++n)
        check(kvlab::detail::enqueue_validated(args, stream, variant));
      check(cudaEventRecord(stop.event, stream));
      check(cudaEventSynchronize(stop.event));
      float elapsed_ms = 0;
      check(cudaEventElapsedTime(&elapsed_ms, start.event, stop.event));

      const auto begin = std::chrono::steady_clock::now();
      check(checked(args, stream));
      check(cudaStreamSynchronize(stream));
      const double ready_us = std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - begin).count();
      std::cout << std::setprecision(10) << "{\"dtype\":\"float16\",\"rows\":" << rows
                << ",\"width\":" << width << ",\"pair\":" << pair << ",\"position\":" << position
                << ",\"implementation\":\"" << (fused ? "fused" : "unfused")
                << "\",\"warmup_calls\":" << warmup << ",\"group_calls\":" << group_calls
                << ",\"event_group_ms\":" << elapsed_ms << ",\"host_ready_us\":" << ready_us << "}\n";
    }
  }
}
}  // namespace

void run_paired_benchmark() {
  check(cudaSetDevice(0));
  for (int rows : {1, 32, 128})
    for (int width : {128, 1024, 4096})
      time_shape(rows, width);
}
