#include "rmsnorm_launch.hpp"

#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cstdint>

namespace kvlab {
namespace {
constexpr int kAddThreads = 256;
constexpr int kNormThreads = 128;
constexpr float kEpsilon = 1e-6F;

template <typename T> __device__ float load_value(T value);
template <> __device__ float load_value(float value) { return value; }
template <> __device__ float load_value(__half value) { return __half2float(value); }
template <typename T> __device__ T store_value(float value);
template <> __device__ float store_value<float>(float value) { return value; }
template <> __device__ __half store_value<__half>(float value) { return __float2half_rn(value); }

template <typename T>
__global__ void residual_add(const T* x, const T* residual, T* z, std::uint64_t count) {
  const auto stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    z[i] = store_value<T>(__fadd_rn(load_value(x[i]), load_value(residual[i])));
  }
}

template <typename T>
__device__ void finish_row(const T* z, const T* weight, T* y, int width,
                           std::uint64_t row, float sum, float* sums) {
  // Even lanes beyond an odd/short row contribute zero and reach every barrier.
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = kNormThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset)
      sums[threadIdx.x] = __fadd_rn(sums[threadIdx.x], sums[threadIdx.x + offset]);
    __syncthreads();
  }
  const float inverse = rsqrtf(__fadd_rn(sums[0] / static_cast<float>(width), kEpsilon));
  for (int j = threadIdx.x; j < width; j += kNormThreads) {
    // Reload the stored rounded residual; do not normalize an unrounded register sum.
    const float scaled = __fmul_rn(load_value(z[row + j]), inverse);
    y[row + j] = store_value<T>(__fmul_rn(scaled, load_value(weight[j])));
  }
}

template <typename T>
__global__ void rmsnorm(const T* z, const T* weight, T* y, int width) {
  __shared__ float sums[kNormThreads];
  const auto row = static_cast<std::uint64_t>(blockIdx.x) * width;
  float sum = 0.0F;
  for (int j = threadIdx.x; j < width; j += kNormThreads) {
    const float value = load_value(z[row + j]);
    sum = __fadd_rn(sum, __fmul_rn(value, value));
  }
  finish_row(z, weight, y, width, row, sum, sums);
}

template <typename T>
__global__ void fused_rmsnorm(const T* x, const T* residual, const T* weight,
                              T* z, T* y, int width) {
  __shared__ float sums[kNormThreads];
  const auto row = static_cast<std::uint64_t>(blockIdx.x) * width;
  float sum = 0.0F;
  for (int j = threadIdx.x; j < width; j += kNormThreads) {
    const auto i = row + j;
    const T rounded = store_value<T>(__fadd_rn(load_value(x[i]), load_value(residual[i])));
    z[i] = rounded;
    const float value = load_value(rounded);
    sum = __fadd_rn(sum, __fmul_rn(value, value));
  }
  finish_row(z, weight, y, width, row, sum, sums);
}

template <typename T>
cudaError_t launch(const RmsNormArgs& args, cudaStream_t stream, detail::Variant variant) {
  if (variant == detail::Variant::Fused) {
    fused_rmsnorm<<<static_cast<unsigned>(args.rows), kNormThreads, 0, stream>>>(
        static_cast<const T*>(args.x.data), static_cast<const T*>(args.residual.data),
        static_cast<const T*>(args.weight.data), static_cast<T*>(args.z.data),
        static_cast<T*>(args.y.data), static_cast<int>(args.width));
    return cudaGetLastError();
  }
  const auto count = static_cast<std::uint64_t>(args.rows) * args.width;
  const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(
      (count + kAddThreads - 1) / kAddThreads, 65535));
  residual_add<<<blocks, kAddThreads, 0, stream>>>(
      static_cast<const T*>(args.x.data), static_cast<const T*>(args.residual.data),
      static_cast<T*>(args.z.data), count);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  rmsnorm<<<static_cast<unsigned>(args.rows), kNormThreads, 0, stream>>>(
      static_cast<const T*>(args.z.data), static_cast<const T*>(args.weight.data),
      static_cast<T*>(args.y.data), static_cast<int>(args.width));
  return cudaGetLastError();
}
cudaError_t checked_launch(const RmsNormArgs& args, cudaStream_t stream, detail::Variant variant) {
  if (validate_rmsnorm_args(args) != RmsNormValidation::Valid) return cudaErrorInvalidValue;
  if (args.rows == 0) return cudaSuccess;
  int device = -1, stream_device = -1;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaStreamGetDevice(stream, &stream_device);
  if (status != cudaSuccess) return status;
  if (stream_device != device) return cudaErrorInvalidDevice;
  const std::array<const void*, 5> pointers = {
      args.x.data, args.residual.data, args.weight.data, args.z.data, args.y.data};
  for (const void* pointer : pointers) {
    cudaPointerAttributes attributes{};
    status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) return status;
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
      return cudaErrorInvalidDevicePointer;
  }
  // Do not erase an outstanding launch error from earlier caller work.
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  return detail::enqueue_validated(args, stream, variant);
}
}  // namespace

cudaError_t detail::enqueue_validated(const RmsNormArgs& args, cudaStream_t stream, Variant variant) {
  return args.type == RmsNormType::Float16 ? launch<__half>(args, stream, variant)
                                          : launch<float>(args, stream, variant);
}

cudaError_t residual_rmsnorm_unfused(const RmsNormArgs& args, cudaStream_t stream) {
  return checked_launch(args, stream, detail::Variant::Unfused);
}

cudaError_t residual_rmsnorm_fused(const RmsNormArgs& args, cudaStream_t stream) {
  return checked_launch(args, stream, detail::Variant::Fused);
}
}  // namespace kvlab
