#pragma once

#include "kvlab/rmsnorm_contract.hpp"
#include <cuda_runtime_api.h>

namespace kvlab {

// Two kernels on the supplied stream: rounded residual store, then RMSNorm reload.
// Checks descriptors, device allocation type/current-device identity, and stream device.
// Caller guarantees contiguous FP16/FP32 storage, truthful byte capacities, writable
// outputs, finite |x|,|residual|<=1024 and |weight|<=16, and input readiness on stream.
// No value scans, allocation, ownership transfer, or completion synchronization.
// Keep all five buffers alive and unmodified until stream completion, including after
// an error: a later launch failure can occur after the residual kernel was enqueued.
// Empty rows: shape/type checked host no-op; no CUDA API calls or buffer access.
// cudaSuccess means accepted/enqueued, NOT completed. Observe execution errors at a
// caller-owned event/stream synchronization boundary. Not an autograd/framework binding.
cudaError_t residual_rmsnorm_unfused(const RmsNormArgs& args, cudaStream_t stream);

}  // namespace kvlab
