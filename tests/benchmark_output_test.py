"""Exercise the executable's output contract and failure paths, without timing gates."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(binary, destination, *extra, succeeds=True):
    command = [binary, "--output", str(destination), "--events", "257", "--seed", "7",
               "--pool-tokens", "64", "--page-tokens", "4", "--slots", "8",
               "--max-tokens", "33", "--prefix-tokens", "8", "--sample-every", "16", *extra]
    result = subprocess.run(command, text=True, capture_output=True)
    assert (result.returncode == 0) == succeeds, (command, result.stdout, result.stderr)


with tempfile.TemporaryDirectory(prefix="kvlab-output-test-") as directory:
    root = Path(directory)
    traces = []
    for share in (0, 90):
        destination = root / str(share)
        run(sys.argv[1], destination, "--share-percent", str(share))
        summary = json.loads((destination / "summary.json").read_text())
        assert summary["schema_version"] == 2 and summary["aggregate_replay_ns"] >= 0
        assert abs(summary["aggregate_ns_per_offered_event"] * 257 - summary["aggregate_replay_ns"]) < 0.001
        control = list(csv.DictReader((destination / "timer-control.csv").open()))
        assert sum(int(row["count"]) for row in control) == 10000
        events = list(csv.DictReader((destination / "events.csv").open()))
        memory = list(csv.DictReader((destination / "memory.csv").open()))
        assert len(events) == 257 == sum(summary["outcomes"].values())
        assert summary["free_pages_after_drain"] == 16
        assert summary["request_metadata"]["peak_live_bytes"] >= summary["request_metadata"]["live_bytes_before_drain"]
        assert sum(group["count"] for group in summary["latency_by_operation_and_outcome"]) == 257
        for group in summary["latency_by_operation_and_outcome"]:
            times = sorted(int(row["elapsed_ns"]) for row in events
                           if (row["kind"], row["outcome"], int(row["metadata_allocations"]) != 0) ==
                           (group["operation"], group["outcome"], group["metadata_allocated"]))
            assert group["p50_ns"] == times[(len(times) * 50 + 99) // 100 - 1]
            assert group["p95_ns"] == times[(len(times) * 95 + 99) // 100 - 1]
            assert group["p99_ns"] is None  # Too few individual observations.
            assert group["zero_duration_samples"] == times.count(0)
        for row in memory:
            value = {k: int(v) for k, v in row.items() if k != "phase"}
            assert value["pool_pages"] == value["live_pages"] + value["free_pages"]
            assert value["live_capacity_tokens"] == (value["unique_committed_tokens"] +
                                                       value["reserved_uncommitted_tokens"] + value["tail_slack_tokens"])
            assert value["modeled_pool_bytes"] == 64 * 2048
        assert memory[-1]["phase"] == "drained" and int(memory[-1]["live_pages"]) == 0
        traces.append([{k: row[k] for k in ("kind", "slot", "tokens", "share_draw")} for row in events])
        original = (destination / "summary.json").read_bytes()
        run(sys.argv[1], destination, succeeds=False)
        assert original == (destination / "summary.json").read_bytes()
    assert traces[0] == traces[1], "policy sweep changed offered trace"
    for args in [("--page-tokens", "0"), ("--page-tokens", "3"), ("--events", "-1"),
                 ("--events", "2junk"), ("--share-percent", "101"), ("--unknown", "1")]:
        run(sys.argv[1], root / "invalid", *args, succeeds=False)
        assert not (root / "invalid").exists()
print("Benchmark JSON/CSV, percentiles, trace pairing, cleanup, and CLI failure checks passed")
