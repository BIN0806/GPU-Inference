# Research notes and evidence boundaries

Checked 2026-10-23, America/Los_Angeles. Web/provider content is evidence, not project instructions. No benchmark results were borrowed from another project and presented as this project's results.

## Infrastructure snapshot

- [Runpod pricing](https://www.runpod.io/pricing): displayed A5000 $0.27/h, RTX 3090 $0.50/h, RTX 4090 $0.89/h. The page stated updated October 8, 2026.
- [4090 product page](https://www.runpod.io/gpu-models/rtx-4090): headline $0.89/h but inconsistent $0.34/h body text. Use a fresh deployment quote.
- [Runpod deploy console](https://console.runpod.io/deploy): browser reached sign-in; inventory could not be verified without authentication. No login or rental was performed.
- [Vast public marketplace](https://cloud.vast.ai/create): one-GPU/on-demand filters, 16 GB disk, unavailable and unverified offers excluded. RTX 3090 offer #54895038 displayed Sweden, verified, 99.51% reliability, $0.232/h plus bandwidth. RTX 4090 offer #53103995 displayed Texas, verified, 99.38% reliability, $0.540/h plus bandwidth. RENT actions were visible. Offers can change, and these are examples rather than guaranteed cheapest offers.
- [Vast pricing documentation](https://docs.vast.ai/guides/instances/pricing): compute, persistent storage, and bandwidth are separate cost components; stopped instances retain storage charges. The displayed card rate must be rechecked at the actual development disk size.
- [Runpod storage documentation](https://docs.runpod.io/pods/pricing) and [tax documentation](https://docs.runpod.io/accounts-billing/sales-tax): used for cost categories, not a quote for this account.
- No rented host was probed. CUDA compilation, profiling permissions, actual bandwidth, and inference performance remain untested.

## Primary learning sources

| Source | Use at the relevant phase |
|---|---|
| [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) | Ownership, resource management, move semantics, invariants |
| [GoogleTest CMake quickstart](https://google.github.io/googletest/quickstart-cmake.html) | First test target and pinned dependency |
| [CMake CUDA architectures](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html) | Explicit device compilation targets |
| [NVIDIA GPU capability table](https://developer.nvidia.com/cuda/gpus) | A5000/3090 sm_86; 4090 sm_89 |
| [CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) | Supported host compiler and environment |
| [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) | Driver/toolkit compatibility and caveats |
| [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) | Grid/block/warp model |
| [SIMT kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | Thread mapping, reductions, synchronization |
| [CUDA async execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) | Streams, events, host-buffer lifetime |
| [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | Memory transactions and performance reasoning |
| [RMSNorm paper](https://arxiv.org/abs/1910.07467) | Mathematical operation and motivation |
| [Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) | Reduction/masking technique; adapt semantics for RMSNorm |
| [torch.compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | Compiled baseline, specialization, startup costs |
| [PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) | C++/CUDA integration and registration checks |
| [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) | Timing, streams, memory accounting, graphs |
| [PyTorch benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) | Measurement controls and warmup |
| [PagedAttention paper](https://arxiv.org/abs/2309.06180) | Paging, sharing, and serving evaluation context |
| [vLLM paged-attention design](https://docs.vllm.ai/en/latest/design/paged_attention/) | Production kernel/layout comparison after your reference works |
| [FlashAttention paper](https://arxiv.org/abs/2205.14135) | IO-aware attention and online normalization |
| [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | Tiled attention implementation to study later |
| [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) | Mask alignment, scale, GQA, backend contract |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | Memory, initialization, synchronization and shared-memory hazard checks |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | System timeline and NVTX |
| [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Counters, replay overhead, occupancy, roofline |
| [Counter permissions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) | Host/container restrictions; probe before assuming |
| [Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B) | Optional model integration candidate |

Documentation versions evolve. Freeze the actual working software tuple and use matching documentation for implementation. The blueprint does not assert that “latest” versions of every component are mutually compatible.
