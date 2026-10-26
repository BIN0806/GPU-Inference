// Standalone rental preflight, not the RMSNorm kernel or a performance benchmark.
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void kvlab_smoke(float* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = static_cast<float>(2 * index + 1);
}

struct DeviceBuffer {
  float* data = nullptr;
  ~DeviceBuffer() { if (data != nullptr) cudaFree(data); }
};

#define CUDA_CHECK(call) do { \
  const auto status = (call); \
  if (status != cudaSuccess) { \
    std::cerr << #call << ": " << cudaGetErrorString(status) << '\n'; \
    return 1; \
  } \
} while (false)

int main() {
  constexpr int count = 257;  // Partial final block exercises the bounds mask.
  DeviceBuffer output;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output.data), count * sizeof(float)));
  kvlab_smoke<<<3, 128>>>(output.data, count);
  CUDA_CHECK(cudaGetLastError());       // Launch configuration errors.
  CUDA_CHECK(cudaDeviceSynchronize()); // Asynchronous execution errors.
  std::vector<float> host(count);
  CUDA_CHECK(cudaMemcpy(host.data(), output.data, count * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < count; ++i) {
    if (host[i] != static_cast<float>(2 * i + 1)) {
      std::cerr << "incorrect element " << i << '\n';
      return 1;
    }
  }
  auto* allocation = output.data;
  output.data = nullptr;
  CUDA_CHECK(cudaFree(allocation));
  std::cout << "KVLAB_NATIVE_CUDA_OK\n";
  return 0;
}
