#include "kvlab/rmsnorm_cuda.hpp"
#include "cuda_check_utils.cuh"
#include "rmsnorm_cuda_bench.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using kvlab::test::check;
using kvlab::test::DeviceBuffer;
using kvlab::test::Stream;
using Operation = decltype(&kvlab::residual_rmsnorm_unfused);
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename T> float value(T x) { return static_cast<float>(x); }
template <typename T> T storage(double x) { return static_cast<T>(x); }
template <> __half storage<__half>(double x) { return __double2half(x); }

template <typename T>
void oracle(const std::vector<T>& x, const std::vector<T>& r, const std::vector<T>& w,
            int rows, int width, std::vector<T>& z, std::vector<T>& y) {
  z.resize(x.size()); y.resize(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float rounded_add = value(x[i]) + value(r[i]);
    z[i] = storage<T>(rounded_add);
  }
  for (int row = 0; row < rows; ++row) {
    double sum = 0;
    const auto base = static_cast<std::size_t>(row) * width;
    for (int j = 0; j < width; ++j) {
      const double v = value(z[base + j]);
      sum += v * v;
    }
    const double denominator = std::sqrt(sum / width + static_cast<double>(1e-6F));
    for (int j = 0; j < width; ++j)
      y[base + j] = storage<T>((value(z[base + j]) / denominator) * value(w[j]));
  }
}

void host_checks(Operation op, const std::string& implementation) {
  // These paths must work without a CUDA device or driver initialization.
  kvlab::RmsNormArgs empty{0, 3, kvlab::RmsNormType::Float16, {}, {}, {}, {}, {}};
  check(op(empty, nullptr));
  empty.width = 0;
  require(op(empty, nullptr) == cudaErrorInvalidValue,
          "invalid empty shape accepted");
  std::vector<__half> x = {storage<__half>(1), storage<__half>(2), storage<__half>(3)};
  std::vector<__half> r = {storage<__half>(std::ldexp(1., -11)),
                         storage<__half>(std::ldexp(1., -10)), storage<__half>(std::ldexp(1., -9))};
  std::vector<__half> w(3, storage<__half>(1)), z, y;
  oracle(x, r, w, 1, 3, z, y);
  require(value(z[2]) == 3.001953125F, "host residual witness failed");
  require(value(y[0]) == 0.462646484375F && value(y[1]) == 0.92529296875F &&
          value(y[2]) == 1.388671875F, "host rounded-normalization witness failed");
  std::vector<float> a{3,4}, b{0,0}, weights{1,1}, fz, fy;
  oracle(a,b,weights,1,2,fz,fy);
  require(fz == a && std::abs(fy[0] - 3/std::sqrt(12.5 + 1e-6F)) < 1e-7,
          "host hand-computed row failed");
  std::cout << "{\"implementation\":\"" << implementation << "\",\"host_checks\":\"passed\",\"device_execution\":false}\n";
}

template <typename T>
void run_case(Operation op, int rows, int width, const std::string& pattern) {
  constexpr bool half = std::is_same_v<T, __half>;
  constexpr auto type = half ? kvlab::RmsNormType::Float16 : kvlab::RmsNormType::Float32;
  std::cerr << "Case: " << (half ? "float16" : "float32") << " M=" << rows
            << " D=" << width << " pattern=" << pattern << '\n';
  const auto count = static_cast<std::size_t>(rows) * width;
  std::vector<T> x(count), r(count), w(width), expected_z, expected_y, z(count), y(count);
  std::mt19937 generator(701);
  std::uniform_real_distribution<float> activation(-3,3), weight(-2,2);
  for (std::size_t i=0; i<count; ++i) {
    x[i] = storage<T>(activation(generator)); r[i] = storage<T>(activation(generator));
    if (pattern == "zero") x[i] = r[i] = storage<T>(0);
    if (pattern == "cancellation") r[i] = storage<T>(-value(x[i]));
    if (pattern == "tiny") { x[i] = storage<T>(value(x[i])*1e-8F); r[i] = storage<T>(value(r[i])*1e-8F); }
    if (pattern == "bounded_large") x[i] = r[i] = storage<T>(1024);
    if (pattern == "rounding") {
      // Repeat the three-channel witness so rounding is not a uniform row rescaling.
      const int channel = static_cast<int>(i % 3);
      x[i] = storage<T>(channel + 1);
      r[i] = storage<T>(std::ldexp(1., (half ? -11 : -24) + channel));
      // Row two isolates using an unrounded add *only* in the denominator.
      if (half && width == 3 && i >= 3)
        r[i] = storage<T>(channel == 0 ? 0.0 : std::ldexp(1., -10));
    }
  }
  for (int j=0; j<width; ++j)
    w[j] = storage<T>(pattern == "rounding" ? 1 : pattern == "bounded_large" ? (j%2 ? 16 : -16) : weight(generator));
  oracle(x,r,w,rows,width,expected_z,expected_y);
  if (rows == 0) {
    check(op({0,width,type,{},{},{},{},{}}, nullptr));
  } else {
    DeviceBuffer<T> dx(count), dr(count), dw(width), dz(count), dy(count);
    Stream stream; // Destroy/drain before the buffers on every exception path.
    check(cudaMemcpyAsync(dx.data,x.data(),dx.bytes,cudaMemcpyHostToDevice,stream.stream));
    check(cudaMemcpyAsync(dr.data,r.data(),dr.bytes,cudaMemcpyHostToDevice,stream.stream));
    check(cudaMemcpyAsync(dw.data,w.data(),dw.bytes,cudaMemcpyHostToDevice,stream.stream));
    kvlab::RmsNormArgs args{rows,width,type,{dx.data,dx.bytes},{dr.data,dr.bytes},
                          {dw.data,dw.bytes},{dz.data,dz.bytes},{dy.data,dy.bytes}};
    auto invalid = args;
    invalid.y.data = invalid.z.data;
    require(op(invalid,stream.stream) == cudaErrorInvalidValue,
            "overlapping output accepted");
    invalid = args; invalid.x.data = x.data();
    require(op(invalid,stream.stream) == cudaErrorInvalidDevicePointer,
            "host pointer accepted");
    check(op(args,stream.stream));
    check(cudaMemcpyAsync(z.data(),dz.data,dz.bytes,cudaMemcpyDeviceToHost,stream.stream));
    check(cudaMemcpyAsync(y.data(),dy.data,dy.bytes,cudaMemcpyDeviceToHost,stream.stream));
    check(cudaStreamSynchronize(stream.stream)); // Observe asynchronous errors here.
  }
  double max_error=0, squared_error=0, squared_reference=0;
  std::size_t worst=0;
  for (std::size_t i=0; i<count; ++i) {
    require(std::memcmp(&z[i],&expected_z[i],sizeof(T)) == 0, "residual bits differ");
    const double actual=value(y[i]), expected=value(expected_y[i]);
    const double error=std::abs(actual-expected);
    if (!std::isfinite(actual) || error > (half ? 2e-3 : 1e-5) + (half ? 2e-3 : 1e-4)*std::abs(expected))
      throw std::runtime_error("normalized output differs at element " + std::to_string(i));
    if (error > max_error) { max_error=error; worst=i; }
    squared_error+=error*error; squared_reference+=expected*expected;
  }
  if (half && rows == 2 && width == 3 && pattern == "rounding") {
    require(value(y[0])==0.462646484375F && value(y[1])==0.92529296875F,
            "rounded-normalization witness failed");
    require(value(y[3]) == 0.462890625F && value(y[4]) == 0.92578125F,
            "RMS denominator used the unrounded residual");
  }
  std::cout << std::setprecision(17) << "{\"dtype\":\"" << (half?"float16":"float32")
            << "\",\"rows\":" << rows << ",\"width\":" << width << ",\"pattern\":\"" << pattern
            << "\",\"residual_bitwise_equal\":true,\"max_abs_error\":" << max_error
            << ",\"normalized_rms_error\":" << std::sqrt(squared_error/std::max(squared_reference,1e-30))
            << ",\"worst_flat_index\":" << (count ? std::to_string(worst) : "null") << "}\n";
}

template <typename T> int matrix(Operation op, bool full) {
  int cases=0;
  const std::vector<int> rows = full ? std::vector<int>{0,1,2,31,32,33,128} : std::vector<int>{0,1,2};
  const std::vector<int> widths = full ? std::vector<int>{1,3,31,32,33,127,128,129,768,1024,4096,8192}
                                       : std::vector<int>{1,3,33,129};
  for (int m:rows) for (int d:widths) { run_case<T>(op,m,d,"random"); ++cases; }
  for (int d: full ? std::vector<int>{1,3,33,129,8192} : std::vector<int>{3})
    for (const char* pattern:{"zero","cancellation","tiny","bounded_large","rounding"}) {
      run_case<T>(op,2,d,pattern); ++cases;
    }
  return cases;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--paired-timing") {
      run_paired_benchmark();
      return 0;
    }
    const std::string mode = argc >= 2 ? argv[1] : "";
    const std::string implementation = argc == 3 ? argv[2] : "unfused";
    if (argc > 3 || (mode != "--host-only" && mode != "--smoke" && mode != "--full") ||
        (implementation != "unfused" && implementation != "fused")) {
      std::cerr << "usage: rmsnorm_cuda_check --host-only|--smoke|--full [unfused|fused]"
                   " or --paired-timing\n";
      return 2;
    }
    const Operation op = implementation == "fused" ? kvlab::residual_rmsnorm_fused
                                                    : kvlab::residual_rmsnorm_unfused;
    host_checks(op, implementation);
    if (mode == "--host-only") return 0;
    int devices=0;
    check(cudaGetDeviceCount(&devices));
    require(devices>0,"GPU checks require a CUDA device; no successful skip");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties,0));
    check(cudaSetDevice(0));
    std::cerr << "Device: " << properties.name << "; compute capability " << properties.major << '.' << properties.minor << '\n';
    const bool full = mode == "--full";
    const int cases=matrix<float>(op, full) + matrix<__half>(op, full);
    std::cout << "{\"implementation\":\"" << implementation << "\",\"status\":\"passed\",\"device_execution\":true,\"cases\":" << cases << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA correctness failed: " << error.what() << '\n';
    return 1;
  }
}
