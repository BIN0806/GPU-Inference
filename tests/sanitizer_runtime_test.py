"""Require a working runtime and recognizable fault reports, not merely a crash."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run_probe(binary, mode, environment, log_dir):
    command = [str(binary), mode]
    try:
        result = subprocess.run(command, env=environment, text=True,
                                capture_output=True, timeout=20)
        record = dict(command=command, returncode=result.returncode,
                      stdout=result.stdout, stderr=result.stderr, timed_out=False)
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        record = dict(command=command, returncode=None, stdout=decode(error.stdout),
                      stderr=decode(error.stderr), timed_out=True)
    (log_dir / f"{mode}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def expected_result(record, mode, signature):
    diagnostic = record["stdout"] + record["stderr"]
    valid_exit = (record["returncode"] == 0 if mode == "clean"
                  else record["returncode"] not in (None, 0))
    return not record["timed_out"] and valid_exit and signature in diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--sanitizers", required=True,
                        choices=("address", "undefined", "address,undefined"))
    parser.add_argument("--log-dir", required=True, type=Path)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # Append mandatory failure behavior; preserve host-specific options such as leak checks.
    for variable in ("ASAN_OPTIONS", "UBSAN_OPTIONS"):
        environment[variable] = environment.get(variable, "") + ":halt_on_error=1"
    cases = [("clean", "kvlab sanitizer clean probe passed")]
    if "address" in args.sanitizers.split(","):
        cases.append(("heap-overflow", "AddressSanitizer: heap-buffer-overflow"))
    if "undefined" in args.sanitizers.split(","):
        cases.append(("signed-overflow", "runtime error: signed integer overflow"))
    for mode, signature in cases:
        record = run_probe(args.binary.resolve(), mode, environment, args.log_dir)
        if not expected_result(record, mode, signature):
            print(f"FAIL {mode}: {json.dumps(record, indent=2)}", file=sys.stderr)
            return 1
        print(f"PASS {mode}: expected exit behavior and diagnostic; log: {args.log_dir / (mode + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
