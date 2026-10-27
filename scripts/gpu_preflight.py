"""Inventory or exercise an existing CUDA host; never provision or alter permissions."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time


class CommandInterrupted(KeyboardInterrupt):
    """The interrupted command has been killed/reaped; retain its partial output."""
    def __init__(self, record):
        super().__init__("command interrupted")
        self.record = record


@contextmanager
def termination_as_interrupt():
    """Let CLI SIGTERM follow the same child cleanup path as Ctrl-C."""
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_command(argv, timeout=30):
    started = time.monotonic()
    record = {"argv": [str(v) for v in argv]}
    try:
        process = subprocess.Popen(record["argv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, errors="replace", start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            record.update(status="exited", returncode=process.returncode, stdout=stdout, stderr=stderr)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            # Stop compiler/profiler descendants too; a killed parent alone can leave GPU work.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            interrupted = isinstance(error, KeyboardInterrupt)
            record.update(status="interrupted" if interrupted else "timed_out",
                          returncode=None, stdout=stdout, stderr=stderr)
            if interrupted:
                record["elapsed_seconds"] = time.monotonic() - started
                raise CommandInterrupted(record) from error
    except FileNotFoundError as error:
        record.update(status="not_installed", returncode=None, stdout="", stderr=str(error))
    except OSError as error:
        record.update(status="launch_error", returncode=None, stdout="", stderr=str(error))
    record["elapsed_seconds"] = time.monotonic() - started
    return record


def probe_passed(record, marker):
    return (record["status"] == "exited" and record["returncode"] == 0
            and marker in record["stdout"] + record["stderr"])


def counter_status(record):
    output = record["stdout"] + record["stderr"]
    if record["status"] == "not_installed":
        return "not_installed"
    if "ERR_NVGPUCTRPERM" in output:
        return "permission_denied"
    if (probe_passed(record, "KVLAB_NATIVE_CUDA_OK") and "kvlab_smoke" in output
            and "no kernels were profiled" not in output.lower()):
        return "verified"
    return "failed"  # Version output, unrelated success, crash, or timeout is not coverage.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-gpu", action="store_true", help="compile and execute on the current host")
    parser.add_argument("--cuda-arch", help="explicit nvcc target, e.g. 86 or 89; required with --run-gpu")
    parser.add_argument("--max-seconds", type=int, default=600, help="overall process budget, including inventory")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("this process-group runner supports Linux/macOS only")
    if args.run_gpu and (not args.cuda_arch or not re.fullmatch(r"[0-9]{2,3}[af]?", args.cuda_arch)):
        parser.error("--run-gpu requires an explicit --cuda-arch such as 86 or 89")
    if not 1 <= args.max_seconds <= 1800:
        parser.error("--max-seconds must be between 1 and 1800")
    root = Path(__file__).resolve().parents[1]
    deadline = time.monotonic() + args.max_seconds
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "mode": "execute" if args.run_gpu else "inventory",
              "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "cuda_arch": args.cuda_arch,
              "max_seconds": args.max_seconds, "python_executable": args.python,
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("tools/gpu_smoke.cu", "scripts/gpu_preflight.py", "scripts/torch_cuda_probe.py")},
              "commands": {},
              "execution_checks_passed": False, "performance_counters": "not_tested",
              "timeline_capture": "not_tested"}
    target = output / "preflight.json"
    def save():
        target.write_text(json.dumps(report, indent=2) + "\n")
    def run(name, argv, timeout=30):
        remaining = deadline - time.monotonic()
        try:
            record = (run_command(argv, min(timeout, remaining)) if remaining > 0 else
                      {"argv": list(map(str, argv)), "status": "timed_out", "returncode": None,
                       "stdout": "", "stderr": "overall preflight deadline exhausted", "elapsed_seconds": 0})
        except CommandInterrupted as error:
            report["commands"][name] = error.record
            save()
            raise
        report["commands"][name] = record
        save()
        return record
    save()
    try:
        run("git", ["git", "-C", root, "rev-parse", "HEAD"])
        run("git_status", ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"])
        for name, command in {
                "gpu_inventory": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
                "nvcc": ["nvcc", "--version"], "host_compiler": ["c++", "--version"],
                "ncu": ["ncu", "--version"], "nsys": ["nsys", "--version"],
                "compute_sanitizer": ["compute-sanitizer", "--version"]}.items():
            run(name, command)
        helper = root / "scripts/torch_cuda_probe.py"
        run("torch_inventory", [args.python, helper, "--mode", "inventory"], 60)
        if not args.run_gpu:
            report["status"] = "inventory_complete"
            return 0
        binary = output / "gpu_smoke"
        built = run("native_build", ["nvcc", "-std=c++17", "-O2", "-lineinfo",
                                     "-arch=sm_" + args.cuda_arch, root / "tools/gpu_smoke.cu",
                                     "-o", binary], 120)
        native_ok = False
        if built["status"] == "exited" and built["returncode"] == 0:
            native = run("native", [binary], 60)
            native_ok = probe_passed(native, "KVLAB_NATIVE_CUDA_OK")
        else:
            report["status"] = "failed_native_build"
            return 2
        if not native_ok:
            report["status"] = "failed_native_execution"
            return 2
        execution = {"native": native_ok}
        for mode in ("torch", "triton", "compile"):
            record = run(mode, [args.python, helper, "--mode", mode], 180 if mode == "compile" else 60)
            execution[mode] = probe_passed(record, "KVLAB_" + mode.upper() + "_CUDA_OK")
        memory = run("memcheck", ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99", binary], 120)
        execution["memcheck"] = (probe_passed(memory, "KVLAB_NATIVE_CUDA_OK") and
                                 "ERROR SUMMARY: 0 errors" in memory["stdout"] + memory["stderr"])
        counters = run("counter_probe", ["ncu", "--set", "basic", "--launch-count", "1", binary], 120)
        report["performance_counters"] = counter_status(counters)
        timeline = run("timeline", ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                                    "-o", output / "timeline", binary], 120)
        report["timeline_capture"] = ("verified" if probe_passed(timeline, "KVLAB_NATIVE_CUDA_OK")
                                      and (output / "timeline.nsys-rep").is_file() else "failed")
        report["execution_checks"] = execution
        report["execution_checks_passed"] = all(execution.values())
        report["status"] = "execution_verified" if report["execution_checks_passed"] else "failed_execution_checks"
        return 0 if report["execution_checks_passed"] else 2
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        return 130
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted_or_failed"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"{report['status']}; counters={report['performance_counters']}; report={target}")


if __name__ == "__main__":
    with termination_as_interrupt():
        raise SystemExit(main())
