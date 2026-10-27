# Unfused CUDA residual + RMSNorm control

Status: source implemented; CPU descriptor tests executed; native CUDA
compile/link and host-only runner checks pending at the implementation commit.
**GPU correctness, Compute Sanitizer, and performance remain untested.** This is
the separate-kernel control for a future fusion experiment, not a fused kernel,
framework binding, or complete transformer block.

## Interfaces and responsibilities

The [CPU descriptor contract](../include/kvlab/rmsnorm_contract.hpp) is independent
of CUDA and PyTorch. `RmsNormArgs` describes contiguous x/r/z/y `[M,D]` buffers,
a weight `[D]` buffer, one shared storage dtype, and byte capacities. Inputs are
const views; outputs are writable views. No object takes ownership.

The native limits are FP16/FP32, `0 <= M <= INT32_MAX`, and `1 <= D <= 8192`.
The M limit reflects one normalization block per row. Width, dtype, capacities,
natural element alignment, address arithmetic, and overlap among all five used
ranges are checked on the host. Adjacent ranges are valid. Unused declared
capacity does not make otherwise disjoint accessed ranges overlap.

An empty batch validates shape and dtype, accepts empty buffers, and returns
without a CUDA call. Unlike the high-level tensor reference, the native no-op
needs no weight tensor; there is no element to read. Invalid nonempty descriptors
return `cudaErrorInvalidValue`; `validate_rmsnorm_args` gives the specific reason.

The [CUDA API](../include/kvlab/rmsnorm_cuda.hpp) then checks the current device,
stream's device, and all five pointers' device allocation type/device identity.
Managed memory and host pointers are outside this first API. Caller-provided
capacities must be truthful; they are not queried allocation extents. The caller
also guarantees finite values and the [existing bounds](RMSNORM.md): activation
magnitudes <=1024, weights <=16. No value scan or silent clamping occurs here.
A future tensor binding must enforce its public value/domain policy separately.

```cpp
// Buffers already allocated and populated on stream; capacities are in bytes.
kvlab::RmsNormArgs args{
    rows, width, kvlab::RmsNormType::Float16,
    {x, activation_bytes}, {residual, activation_bytes}, {weight, weight_bytes},
    {z, activation_bytes}, {y, activation_bytes}};
const cudaError_t enqueued = kvlab::residual_rmsnorm_unfused(args, stream);
// Handle enqueued; observe completion/errors before reusing any buffer.
```

Success means the launches were accepted, not that their work completed. Keep
inputs and outputs alive and unchanged until the supplied stream finishes.
Order all producers before the operation and downstream consumers after it on
that stream, or establish explicit event dependencies. The function performs
no allocation, copy, ownership transfer, or synchronization. Even a returned
error can leave work pending if the add launch succeeded before normalization
failed. Cleanup must drain outstanding work; a failed call is not a rollback.

## Kernel mechanics

[residual_rmsnorm.cu](../src/kernels/residual_rmsnorm.cu) contains two launches:

1. A 256-thread, capped-grid, grid-stride add kernel. Its 64-bit element offsets
   cover the flattened input and it rounds the FP32 sum to storage type in `z`.
2. One 128-thread block per row for RMSNorm. Each thread sums its strided columns
   in FP32, writes a shared partial, and participates in every barrier of a
   binary-tree reduction. Lanes with no columns contribute zero. Divide by the
   actual width, add FP32 epsilon `1e-6`, apply reciprocal square root, then
   reread `z` and write weighted `y` in the original storage dtype.

The block-wide tree is deliberately explicit. Warp shuffles, vectorization,
register retention, dispatch tuning, and fusion would each change a mechanism
we can later isolate. Widths smaller than a block and odd tails cannot skip a
barrier. No `--use_fast_math`; explicit rounded intrinsics and compiler options
preserve the intended arithmetic. Different reduction orders can still produce
small normalized-output differences within the existing tolerance.

The logical traffic of this two-pass control is **7 scalar element transfers
per activation element**: x/r reads and z write, two z reads, weight read, y
write. This is a logical byte model, not measured DRAM traffic; caching can
change physical transactions. Device/pointer checks also add host overhead.
This API is a correctness baseline; a later timing harness must clearly separate
host validation, allocation, and device work. There is no speedup claim.

## Build without requiring a GPU at configure time

On Linux with CUDA Toolkit 12.8+ development tools and a supported host compiler:

```sh
cmake -S . -B build/cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DKVLAB_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/cuda --target rmsnorm_cuda_check --parallel 2
./build/cuda/rmsnorm_cuda_check --host-only
```

Choose the actual target architecture; 86 is an example. An explicit numeric
architecture list is required. CPU builds keep CUDA disabled and need no CUDA
headers or runtime. `kvlab_cuda` links only the CUDA runtime, independently of
the allocator and PyTorch. Host ASan/UBSan settings do not instrument device code.

CUDA 12.8 is selected because the wrapper uses `cudaStreamGetDevice`. CI installs
only hash-pinned NVIDIA 12.8.1 redistributable compiler/runtime/CCCL archives into
a new temporary directory; it installs no driver. [Installer](../scripts/install_cuda_compile_toolchain.sh),
[dependency record](DEPENDENCIES.md), [official manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json).

## Verification modes and evidence boundary

The [native checker](../tests/cuda/rmsnorm_cuda_check.cu) has three explicit modes:

| Mode | What executes | What it establishes |
|---|---|---|
| `--host-only` | Empty/invalid native API paths and FP64 host-oracle golden cases | Linked program and host contracts can run without a device. Output says `device_execution:false`. |
| `--smoke` | 34 cases across FP16/FP32, empty/small/odd shapes and adversarial patterns | Small hardware check, only after real GPU execution. |
| `--full` | 218 cases using the same shape grid and pattern categories as the Python reference | Native GPU numerical matrix, only after real GPU execution. |

The native runner uses its own C++ FP64 oracle with rounded residuals and direct
FP64-to-FP16 conversion. It is separate from the kernel's indexing/reduction,
and its hand-computed and rounding witnesses match the Python oracle. It uses
ordinary double summation, not Python `math.fsum`, and its random values are not
identical to PyTorch's generator. The two 218-case runs are complementary tests,
not identical inputs or interchangeable evidence.

GPU modes create a non-default nonblocking stream, enqueue copies and both
kernels there, copy results back, and synchronize at the checking boundary.
They compare z storage bits, finite y values, existing per-element tolerances,
maximum absolute error, and normalized RMS error. A targeted FP16 case checks
rounded-residual semantics. Active shape/pattern goes to stderr before work;
per-case JSON goes to stdout. The runner rejects overlapping outputs and host
pointers. It does not yet stress multi-device, cross-stream dependency transfer,
or application-level allocator lifetimes. It refuses to pass when no GPU is
available; a compile-only runner must select `--host-only` explicitly.

On an already validated GPU host, save stdout and stderr and preserve exit codes:

```sh
./build/cuda/rmsnorm_cuda_check --smoke > artifacts/cuda-smoke.jsonl 2> artifacts/cuda-smoke.log
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/cuda/rmsnorm_cuda_check --smoke
compute-sanitizer --tool racecheck --error-exitcode 99 ./build/cuda/rmsnorm_cuda_check --smoke
compute-sanitizer --tool synccheck --error-exitcode 99 ./build/cuda/rmsnorm_cuda_check --smoke
./build/cuda/rmsnorm_cuda_check --full > artifacts/cuda-full.jsonl 2> artifacts/cuda-full.log
```

Create `artifacts/` first and choose fresh output names. These commands are
prepared, **not executed hardware results**. Use [GPU preflight](GPU_PREFLIGHT.md)
and [budget/session accounting](BUDGET.md) before a paid session. Compilation
alone does not satisfy the Phase 2 correctness or performance milestone.

References: [NVIDIA stream API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__STREAM.html)
and [block synchronization](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#synchronization-functions).
