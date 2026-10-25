"""Pair two Release allocator binaries; reject mismatched work before reporting costs."""
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
import shlex
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_build(binary):
    cache = (binary.parent / "CMakeCache.txt").read_text()
    if "CMAKE_BUILD_TYPE:STRING=Release\n" not in cache:
        raise ValueError(f"{binary}: expected a Release CMake cache beside the binary")
    commands = json.loads((binary.parent / "compile_commands.json").read_text())
    if not commands:
        raise ValueError("empty compilation database")
    for entry in commands:
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        if any(arg.startswith("-fsanitize=") for arg in arguments):
            raise ValueError(f"{binary}: instrumented build is not a timing baseline")
    return {"path": str(binary), "binary_sha256": digest(binary),
            "cmake_cache_sha256": digest(binary.parent / "CMakeCache.txt"),
            "compile_commands_sha256": digest(binary.parent / "compile_commands.json")}


def event_digest(path):
    result = hashlib.sha256()
    with path.open() as stream:
        for row in csv.DictReader(stream):
            # Timing and PMR calls may differ; offered work AND every outcome must match.
            result.update(json.dumps([row[k] for k in
                                     ("index", "kind", "slot", "tokens", "share_draw", "outcome")]).encode())
    return result.hexdigest()


def compare_outputs(first, second):
    summaries = [json.loads((p / "summary.json").read_text()) for p in (first, second)]
    for key in ("schema_version", "measurement", "config", "kv_model", "warmup_events",
                "outcomes", "live_pages_before_drain", "free_pages_after_drain"):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError(f"baseline/candidate mismatch: {key}")
    if any(s["build_type"] != "Release" or s["schema_version"] != 2 for s in summaries):
        raise ValueError("expected Release schema-2 benchmark output")
    trace = event_digest(first / "events.csv")
    if event_digest(second / "events.csv") != trace:
        raise ValueError("offered events or individual outcomes differ")
    if (first / "memory.csv").read_bytes() != (second / "memory.csv").read_bytes():
        raise ValueError("sampled physical-memory accounting differs")
    return summaries, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--events", type=int, default=20000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--page-tokens", nargs="+", type=int, default=[8, 16, 64])
    parser.add_argument("--sharing", nargs="+", type=int, default=[0, 90])
    parser.add_argument("--order-seed", type=int, default=601)
    args = parser.parse_args()
    if args.events < 1 or args.repeats < 1:
        parser.error("events and repeats must be positive")
    if any(len(values) != len(set(values)) for values in (args.seeds, args.page_tokens, args.sharing)):
        parser.error("seeds, page sizes, and sharing values must be unique")
    if any(p < 1 or 4096 % p for p in args.page_tokens) or any(s < 0 or s > 100 for s in args.sharing):
        parser.error("page sizes must divide 4096; sharing must be 0..100")
    root = Path(__file__).resolve().parents[1]
    binaries = {name: getattr(args, name).resolve(strict=True) for name in ("baseline", "candidate")}
    builds = {name: inspect_build(binary) for name, binary in binaries.items()}
    baseline_sources = json.loads(args.baseline_source_manifest.read_text())
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "--cached", "--others",
                                     "--exclude-standard", "-z"]).decode().split("\0")[:-1]
    candidate_sources = {p: digest(root / p) for p in sorted(set(paths))
                         if Path(p).suffix in (".cpp", ".hpp", ".py") or p in ("CMakeLists.txt", "CMakePresets.json")}
    pairs = list(itertools.product(args.page_tokens, args.sharing, args.seeds, range(args.repeats)))
    rng = random.Random(args.order_seed)
    rng.shuffle(pairs)
    schedule = []
    for page, share, seed, repeat in pairs:
        order = ["baseline", "candidate"]
        rng.shuffle(order)
        schedule.append(dict(page_tokens=page, sharing=share, seed=seed, repeat=repeat, order=order))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, binary in binaries.items():
        for file in ("CMakeCache.txt", "compile_commands.json"):
            (output / f"{name}-{file}").write_bytes((binary.parent / file).read_bytes())
    manifest = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                    platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                    base_commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                    baseline_sources=baseline_sources, candidate_sources=candidate_sources, builds=builds,
                    events=args.events, order_seed=args.order_seed, schedule=schedule, pairs_completed=0, commands=[])
    (output / "tracked-working-tree.patch").write_bytes(subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "HEAD"]))
    manifest_path = output / "manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    rows = []
    try:
        for index, pair in enumerate(schedule):
            label = f"p{pair['page_tokens']}-s{pair['sharing']}-seed{pair['seed']}-r{pair['repeat']}"
            destinations = {name: output / (label + "-" + name) for name in binaries}
            for name in pair["order"]:
                command = [str(binaries[name]), "--output", str(destinations[name]), "--events", str(args.events),
                           "--page-tokens", str(pair["page_tokens"]), "--share-percent", str(pair["sharing"]),
                           "--seed", str(pair["seed"]), "--pool-tokens", "4096", "--slots", "32",
                           "--max-tokens", "1024", "--prefix-tokens", "128", "--sample-every", "128"]
                manifest["commands"].append(dict(argv=command, load_average_before=os.getloadavg() if hasattr(os, "getloadavg") else None))
                save()
                # Direct file output preserves diagnostics even if the bounded child times out.
                with (output / (label + "-" + name + ".log")).open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
            summaries, trace = compare_outputs(destinations["baseline"], destinations["candidate"])
            row = dict(pair=index, **{k: v for k, v in pair.items() if k != "order"},
                       first=pair["order"][0], trace_and_outcomes_sha256=trace)
            for name, summary in zip(("baseline", "candidate"), summaries):
                row[name + "_ns_per_event"] = summary["aggregate_ns_per_offered_event"]
                for key in ("allocation_calls", "requested_bytes", "peak_live_bytes", "live_bytes_after_drain"):
                    row[name + "_" + key] = summary["request_metadata"][key]
            row["candidate_over_baseline_time"] = (row["candidate_ns_per_event"] / row["baseline_ns_per_event"]
                                                    if row["baseline_ns_per_event"] else "")
            rows.append(row)
            with (output / "pairs.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            manifest["pairs_completed"] = index + 1
            save()
            print(f"{index + 1}/{len(schedule)} {label}: matched", flush=True)
        for name, binary in binaries.items():
            if digest(binary) != builds[name]["binary_sha256"]:
                raise ValueError(f"{name} binary changed during the experiment")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
