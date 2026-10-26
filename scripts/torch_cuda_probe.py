"""Actual execution probes for an already provisioned host; no resource provisioning."""
import argparse
import importlib.util
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inventory", "torch", "triton", "compile"), required=True)
    args = parser.parse_args()
    if importlib.util.find_spec("torch") is None:
        print(json.dumps({"torch_installed": False}))
        return 0 if args.mode == "inventory" else 1
    import torch
    if args.mode == "inventory":
        print(json.dumps({"torch_installed": True, "torch": torch.__version__,
                          "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                          "triton_installed": importlib.util.find_spec("triton") is not None}))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA device")
    x = torch.arange(257, device="cuda", dtype=torch.float32)
    if args.mode == "torch":
        output = x * 2 + 1
    elif args.mode == "compile":
        # Trigger compilation, launch, synchronization and value validation.
        operation = torch.compile(lambda values: values * 2 + 1, fullgraph=True)
        output = operation(x)
    else:
        import triton
        import triton.language as tl
        @triton.jit
        def affine(values, output, count: tl.constexpr, BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < count
            data = tl.load(values + offsets, mask=mask, other=0)
            tl.store(output + offsets, data * 2 + 1, mask=mask)
        output = torch.empty_like(x)
        affine[(triton.cdiv(x.numel(), 128),)](x, output, x.numel(), BLOCK=128)
    torch.cuda.synchronize()
    expected = torch.arange(257, dtype=torch.float32) * 2 + 1
    if not torch.equal(output.cpu(), expected):
        raise RuntimeError("CUDA probe produced incorrect values")
    print("KVLAB_" + args.mode.upper() + "_CUDA_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
