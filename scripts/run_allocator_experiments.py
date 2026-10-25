"""Serial, randomized-order fixed-budget experiments; Python standard library only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=20000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--page-tokens", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--sharing", type=int, nargs="+", default=[0, 50, 90])
    parser.add_argument("--pool-tokens", type=int, default=4096)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--sample-every", type=int, default=128)
    parser.add_argument("--order-seed", type=int, default=123)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    if args.repeats < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("repeats must be positive and seeds must be unique")
    cache_path = binary.parent / "CMakeCache.txt"
    cache = cache_path.read_text()
    if "CMAKE_BUILD_TYPE:STRING=Release\n" not in cache or "-fsanitize=" in cache:
        parser.error("experiment runner requires a Release build without sanitizers")
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "--cached", "--others",
                                     "--exclude-standard", "-z"]).decode().split("\0")[:-1]
    sources = {p: digest(root / p) for p in sorted(set(paths))
               if Path(p).suffix in (".cpp", ".hpp", ".py") or p in ("CMakeLists.txt", "CMakePresets.json")}
    jobs = list(itertools.product(args.page_tokens, args.sharing, args.seeds, range(args.repeats)))
    random.Random(args.order_seed).shuffle(jobs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
        "binary": str(binary), "binary_sha256": digest(binary), "source_sha256": sources,
        "base_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_manifest_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "jobs": jobs, "commands": [], "runs_completed": 0,
        "limitations": ["CPU metadata only; no KV payload or GPU memory allocated",
                        "No pinned core, clock control, thermal measurement, or exclusive host",
                        "Individual event timers and PMR counters perturb small operations",
                        "Sampled memory peaks can miss unsampled transients"],
    }
    (output / "CMakeCache.txt").write_text(cache)
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "HEAD"])
    (output / "tracked-working-tree.patch").write_bytes(diff)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    traces_by_seed = {}
    rows = []
    try:
        for index, (page, sharing, seed, repeat) in enumerate(jobs):
            run_name = f"p{page}-share{sharing}-seed{seed}-r{repeat}"
            destination = output / run_name
            command = [str(binary), "--output", str(destination), "--events", str(args.events),
                       "--seed", str(seed), "--page-tokens", str(page), "--share-percent", str(sharing),
                       "--pool-tokens", str(args.pool_tokens), "--slots", str(args.slots),
                       "--max-tokens", str(args.max_tokens), "--prefix-tokens", str(args.prefix_tokens),
                       "--sample-every", str(args.sample_every)]
            manifest["commands"].append({"argv": command, "load_average_before": os.getloadavg() if hasattr(os, "getloadavg") else None})
            result = subprocess.run(command, text=True, capture_output=True)
            (output / (run_name + ".log")).write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"{run_name} exited {result.returncode}; retained its log")
            summary = json.loads((destination / "summary.json").read_text())
            if summary["build_type"] != "Release":
                raise RuntimeError("binary reports a non-Release configuration")
            with (destination / "events.csv").open() as stream:
                trace_hash = hashlib.sha256()
                for event in csv.DictReader(stream):
                    trace_hash.update(json.dumps([event[key] for key in
                                                 ("kind", "slot", "tokens", "share_draw")]).encode())
            trace_sha = trace_hash.hexdigest()
            if traces_by_seed.setdefault(seed, trace_sha) != trace_sha:
                raise RuntimeError("offered trace changed within a paired seed")
            with (destination / "memory.csv").open() as stream:
                memory = [{k: int(v) if k != "phase" else v for k, v in row.items()}
                          for row in csv.DictReader(stream) if row["phase"] == "sample"]
            outcomes = summary["outcomes"]
            attempts = outcomes["admitted"] + outcomes["admission_rejected"]
            row = {"run": run_name, "page_tokens": page, "sharing": sharing, "seed": seed, "repeat": repeat,
                   **outcomes, "admission_rejection_fraction": outcomes["admission_rejected"] / attempts if attempts else 0,
                   "sampled_mean_occupancy": statistics.mean(m["live_pages"] / m["pool_pages"] for m in memory),
                   "sampled_mean_tail_slack_tokens": statistics.mean(m["tail_slack_tokens"] for m in memory),
                   "sampled_mean_cached_only_pages": statistics.mean(m["cached_only_pages"] for m in memory),
                   "sampled_peak_live_pages": max(m["live_pages"] for m in memory),
                   "metadata_allocation_calls": summary["request_metadata"]["allocation_calls"],
                   "metadata_peak_live_bytes": summary["request_metadata"]["peak_live_bytes"],
                   "aggregate_ns_per_offered_event": summary["aggregate_ns_per_offered_event"],
                   "timer_control_zero_samples": summary["timer_control_zero_samples"],
                   "timer_control_p50_ns": summary["timer_control_p50_ns"], "trace_sha256": trace_sha}
            for allocated in (False, True):
                group = next((g for g in summary["latency_by_operation_and_outcome"]
                              if g["outcome"] == "appended" and g["metadata_allocated"] == allocated), None)
                label = "append_with_metadata_allocation" if allocated else "append_without_metadata_allocation"
                row[label + "_p50_ns"] = group["p50_ns"] if group else ""
                row[label + "_p95_ns"] = group["p95_ns"] if group else ""
                row[label + "_samples"] = group["count"] if group else 0
                row[label + "_zero_duration_samples"] = group["zero_duration_samples"] if group else 0
            rows.append(row)
            with (output / "runs.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest["runs_completed"] = index + 1
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"{index + 1}/{len(jobs)} {run_name}", flush=True)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
