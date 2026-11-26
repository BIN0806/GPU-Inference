#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <stdexcept>

namespace kvlab::test {
inline void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T> struct DeviceBuffer {
  T* data = nullptr;
  std::size_t bytes;
  explicit DeviceBuffer(std::size_t count) : bytes(count * sizeof(T)) {
    if (bytes) check(cudaMalloc(reinterpret_cast<void**>(&data), bytes));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { if (data) cudaFree(data); }
};
struct Stream {
  cudaStream_t stream{};
  Stream() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() {
    // The runner drains on exceptional paths before its buffers are destroyed.
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
};
}  // namespace kvlab::test
