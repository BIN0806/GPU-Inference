"""Execute the shape/dtype matrix against the independent FP64 oracle; no timings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import torch
from kvlab.correctness import ROWS, WIDTHS, check_case, make_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "device": args.device, "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
              "python": platform.python_version(), "platform": platform.platform(),
              "base_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((root / "python/kvlab").glob("*.py"))}, "cases": []}
    destination = args.output / "correctness.json"
    report["source_sha256"]["scripts/check_rmsnorm_reference.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        for dtype in (torch.float16, torch.float32):
            cases = [(m, d, "random") for m in ROWS for d in WIDTHS]
            cases += [(2, d, pattern) for d in (1, 3, 33, 129, 8192)
                      for pattern in ("zero", "cancellation", "tiny", "bounded_large", "rounding")]
            for rows, width, pattern in cases:
                report["active_case"] = {"dtype": str(dtype), "rows": rows, "width": width,
                                         "pattern": pattern, "seed": 701}
                save()
                record = check_case(*make_case(rows, width, dtype, pattern, device=args.device))
                report["cases"].append({"pattern": pattern, "seed": 701, **record})
                save()
        report["status"] = "passed"
        report.pop("active_case", None)
        print(f"Passed {len(report['cases'])} cases on {args.device}; report: {destination}")
        for dtype in ("float16", "float32"):
            cases = [r for r in report["cases"] if r["dtype"] == dtype]
            print(json.dumps({"dtype": dtype, "cases": len(cases),
                              "max_abs_error": max(r["max_abs_error"] for r in cases),
                              "max_normalized_rms_error": max(r["normalized_rms_error"] for r in cases)}))
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
