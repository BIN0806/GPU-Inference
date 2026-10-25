"""Plot paired CPU results without turning repeats into confidence intervals."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with (args.directory / "pairs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    events = json.loads((args.directory / "manifest.json").read_text())["events"]
    groups = sorted({(int(r["page_tokens"]), int(r["sharing"])) for r in rows})
    selected = [[r for r in rows if (int(r["page_tokens"]), int(r["sharing"])) == key] for key in groups]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), layout="constrained")
    for ax, metric, title, scale in (
            (axes[0], "allocation_calls", f"Host table allocations / {events:,} offered events", 1),
            (axes[1], "live_bytes_after_drain", "Retained host metadata after drain (KiB)", 1024)):
        for variant, offset, color in (("baseline", -0.18, "#64748b"), ("candidate", 0.18, "#0d9488")):
            values = [[float(r[f"{variant}_{metric}"]) / scale for r in group] for group in selected]
            centers = [statistics.median(v) for v in values]
            errors = [[c - min(v) for c, v in zip(centers, values)], [max(v) - c for c, v in zip(centers, values)]]
            ax.bar([i + offset for i in range(len(groups))], centers, width=0.34,
                   yerr=errors, capsize=3, label=variant, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(bottom=0)
    for i, group in enumerate(selected):
        ratios = [float(r["candidate_over_baseline_time"]) for r in group]
        positions = [i + (j - (len(ratios) - 1) / 2) * 0.035 for j in range(len(ratios))]
        axes[2].scatter(positions, ratios, s=18, color="#0d9488", alpha=0.7)
        axes[2].plot([i - 0.24, i + 0.24], [statistics.median(ratios)] * 2, color="#0f172a", linewidth=2)
    axes[2].axhline(1, color="#b91c1c", linestyle="--", linewidth=1)
    axes[2].set_title("Paired whole-trace time ratio\ncandidate / baseline; below 1 is faster", fontsize=11)
    for ax in axes:
        ax.set_xticks(range(len(groups)), [f"P={p}\nshare={s}%" for p, s in groups], fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle("Request-table storage reuse: fewer allocations, more retained memory\n"
                 "CPU metadata only · 3 trace seeds × 3 repeats per case · medians and observed ranges", fontsize=13)
    fig.savefig(args.directory / "request-growth-comparison.png", dpi=160)
    fig.savefig(args.directory / "request-growth-comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
