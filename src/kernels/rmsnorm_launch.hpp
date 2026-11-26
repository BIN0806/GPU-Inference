#pragma once
#include "kvlab/rmsnorm_cuda.hpp"

namespace kvlab::detail {
enum class Variant { Unfused, Fused };

// Internal launch path used by the public wrappers and the stream timing loop.
// Requires a nonempty descriptor already checked on the same device and stream.
cudaError_t enqueue_validated(const RmsNormArgs& args, cudaStream_t stream, Variant variant);
}  // namespace kvlab::detail
