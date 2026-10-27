"""Build and validate the native control on an existing host; never rent a GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import time

from gpu_preflight import CommandInterrupted, run_command, termination_as_interrupt


SOURCE_FILES = (
    "CMakeLists.txt", "include/kvlab/rmsnorm_contract.hpp", "include/kvlab/rmsnorm_cuda.hpp",
    "src/kernels/residual_rmsnorm.cu", "tests/cuda/rmsnorm_cuda_check.cu",
    "scripts/check_rmsnorm_cuda.py", "scripts/gpu_preflight.py",
)
HOST_RESULT = {"host_checks": "passed", "device_execution": False}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exited_ok(record):
    return record["status"] == "exited" and record["returncode"] == 0


def expected_cases(mode):
    rows = [0, 1, 2, 31, 32, 33, 128] if mode == "full" else [0, 1, 2]
    widths = [1, 3, 31, 32, 33, 127, 128, 129, 768, 1024, 4096, 8192] if mode == "full" else [1, 3, 33, 129]
    edge_widths = [1, 3, 33, 129, 8192] if mode == "full" else [3]
    cases = {(dtype, m, d, "random") for dtype in ("float16", "float32") for m in rows for d in widths}
    cases.update((dtype, 2, d, pattern) for dtype in ("float16", "float32") for d in edge_widths
                 for pattern in ("zero", "cancellation", "tiny", "bounded_large", "rounding"))
    return cases


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def checker_passed(record, mode):
    """A success banner cannot substitute for the full, unique numerical case set."""
    if not exited_ok(record):
        return False
    try:
        lines = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
                 for line in record["stdout"].splitlines() if line.strip()]
        if not lines or lines[0] != HOST_RESULT or lines[0].get("device_execution") is not False:
            return False
        if mode == "host-only":
            return len(lines) == 1
        expected = expected_cases(mode)
        if (lines[-1] != {"status": "passed", "device_execution": True, "cases": len(expected)}
                or lines[-1].get("device_execution") is not True):
            return False
        seen = set()
        for row in lines[1:-1]:
            if type(row["rows"]) is not int or type(row["width"]) is not int:
                return False
            key = (row["dtype"], row["rows"], row["width"], row["pattern"])
            if key not in expected or key in seen or row["residual_bitwise_equal"] is not True:
                return False
            for field in ("max_abs_error", "normalized_rms_error"):
                value = row[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    return False
            index = row["worst_flat_index"]
            count = row["rows"] * row["width"]
            if (count == 0 and index is not None) or (count > 0 and (type(index) is not int or not 0 <= index < count)):
                return False
            seen.add(key)
        return seen == expected
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return False


def sanitizer_clean(text, tool):
    # Unknown formats fail closed. A zero summary must not hide another nonzero one.
    if re.search(r"internal sanitizer error|no attachable process|fatal", text, re.IGNORECASE):
        return False
    summaries = [line.strip().lstrip("=").strip() for line in text.splitlines() if "SUMMARY:" in line]
    expected = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                if tool == "racecheck" else "ERROR SUMMARY: 0 errors")
    return summaries == [expected]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--host-only", action="store_true", help="compile and check host paths; no GPU proof")
    mode.add_argument("--run-gpu", action="store_true", help="run numerical and sanitizer gates on an existing GPU")
    parser.add_argument("--cuda-arch", required=True, help="one numeric target, e.g. 86 or 89")
    parser.add_argument("--cuda-compiler", type=Path, help="optional absolute path to nvcc")
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("Linux/macOS process-group runner required")
    if not re.fullmatch(r"[0-9]{2,3}", args.cuda_arch):
        parser.error("--cuda-arch must be a single numeric architecture")
    if not 1 <= args.max_seconds <= 1800:
        parser.error("--max-seconds must be between 1 and 1800")
    if args.cuda_compiler and not args.cuda_compiler.is_absolute():
        parser.error("--cuda-compiler must be absolute")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build = output / "build"
    target = output / "validation.json"
    deadline = time.monotonic() + args.max_seconds
    report = {"schema_version": 1, "mode": "host-only" if args.host_only else "gpu",
              "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "cuda_arch": args.cuda_arch,
              "max_seconds": args.max_seconds, "hardware_checks_passed": False,
              "source_sha256": {name: sha256(root / name) for name in SOURCE_FILES},
              "commands": {}, "checks": {}}

    def save():
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)

    def run(name, command, timeout=30):
        report["active_command"] = name
        save()
        remaining = deadline - time.monotonic()
        try:
            record = (run_command(command, min(timeout, remaining)) if remaining > 0 else
                      {"argv": list(map(str, command)), "status": "timed_out", "returncode": None,
                       "stdout": "", "stderr": "overall deadline exhausted", "elapsed_seconds": 0})
        except CommandInterrupted as error:
            report["commands"][name] = error.record
            save()
            raise
        report["commands"][name] = record
        save()
        return record

    def require(name, passed):
        report["checks"][name] = passed
        if not passed:
            report["status"] = "failed_" + name
            raise ValueError("required check failed: " + name)
        save()

    save()
    try:
        for name, command in {
            "git": ["git", "-C", root, "rev-parse", "HEAD"],
            "git_status": ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            "cmake_version": ["cmake", "--version"], "ninja_version": ["ninja", "--version"],
            "nvcc_version": [args.cuda_compiler or "nvcc", "--version"],
        }.items():
            run(name, command)
        configure = ["cmake", "-S", root, "-B", build, "-G", "Ninja", "-DBUILD_TESTING=OFF",
                     "-DCMAKE_BUILD_TYPE=Release", "-DKVLAB_ENABLE_CUDA=ON",
                     "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch]
        if args.cuda_compiler:
            configure.append("-DCMAKE_CUDA_COMPILER=" + str(args.cuda_compiler))
        require("configure", exited_ok(run("configure", configure, 120)))
        require("build", exited_ok(run("build", ["cmake", "--build", build, "--target", "rmsnorm_cuda_check", "--parallel", "2"], 180)))
        binary = build / "rmsnorm_cuda_check"
        report["binary_sha256"] = sha256(binary)
        require("host", checker_passed(run("host", [binary, "--host-only"]), "host-only"))
        if args.run_gpu:
            run("gpu_inventory", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"])
            run("sanitizer_version", ["compute-sanitizer", "--version"])
            require("smoke", checker_passed(run("smoke", [binary, "--smoke"], 60), "smoke"))
            for tool in ("memcheck", "racecheck", "synccheck"):
                log = output / (tool + ".log")
                # Compute Sanitizer expands percent escapes even in an argv path.
                command = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                           "--print-level", "info", "--log-file", str(log).replace("%", "%%"), binary, "--smoke"]
                record = run(tool, command, 120)
                text = log.read_text() if log.is_file() else ""
                report["commands"][tool]["sanitizer_log"] = text
                require(tool, checker_passed(record, "smoke") and sanitizer_clean(text, tool))
            require("full", checker_passed(run("full", [binary, "--full"], 120), "full"))
        require("sources_unchanged", report["source_sha256"] == {name: sha256(root / name) for name in SOURCE_FILES})
        require("binary_unchanged", report["binary_sha256"] == sha256(binary))
        report["hardware_checks_passed"] = args.run_gpu
        report["status"] = "hardware_verified" if args.run_gpu else "host_verified"
        return 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        return 130
    except (OSError, ValueError) as error:
        if report["status"] == "running":
            report["status"] = "failed_runner"
        report["error"] = str(error)
        return 2
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted_or_failed"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"{report['status']}; hardware_checks_passed={report['hardware_checks_passed']}; report={target}")


if __name__ == "__main__":
    with termination_as_interrupt():
        raise SystemExit(main())
