"""Summarize one complete native FP16 timing sweep; optionally plot paired ratios."""
import argparse
import json
from pathlib import Path
from statistics import median, quantiles

from check_rmsnorm_cuda import timing_passed


def summarize(rows):
    samples = {}
    for row in rows:
        shape = (row["rows"], row["width"])
        pair = samples.setdefault(shape, {}).setdefault(row["pair"], {})
        pair[row["implementation"]] = (
            1000 * row["event_group_ms"] / row["group_calls"], row["host_ready_us"])
    for shape, by_pair in sorted(samples.items()):
        metrics = []
        for metric in (0, 1):
            controls = [pair["unfused"][metric] for pair in by_pair.values()]
            candidates = [pair["fused"][metric] for pair in by_pair.values()]
            ratios = [a / b for a, b in zip(controls, candidates)]
            q1, _, q3 = quantiles(ratios, n=4, method="inclusive")
            metrics.append((median(controls), median(candidates), median(ratios), q1, q3))
        yield shape, metrics


def plot(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True, layout="constrained")
    for metric, (ax, title) in enumerate(zip(axes, ("CUDA event groups", "Host call through completion"))):
        for i, (_, metrics) in enumerate(results):
            _, _, middle, low, high = metrics[metric]
            ax.errorbar(middle, i, xerr=[[middle - low], [high - middle]],
                        fmt="o", capsize=3, color="#0a6796")
        ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Control / fused time (×)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(results)), [f"{m} × {d}" for (m, d), _ in results])
    axes[0].set_ylabel("Rows × width")
    axes[0].invert_yaxis()
    fig.suptitle("Warm FP16 residual + RMSNorm\nMedian and middle 50% of 12 paired ratios")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="JSON lines from commands.paired_timing.stdout")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    data = args.samples.read_text()
    if not timing_passed(dict(stdout=data, status="exited", returncode=0)):
        parser.error("expected the complete, paired native timing dataset")
    results = list(summarize(json.loads(line) for line in data.splitlines()))
    print("| Rows × width | Stream control / fused (µs) | Paired ratio [IQR] | "
          "Host control / fused (µs) | Paired ratio [IQR] |")
    print("|---|---:|---:|---:|---:|")
    for (m, d), metrics in results:
        cells = [f"{m} × {d}"]
        for control, fused, ratio, low, high in metrics:
            cells.extend((f"{control:.2f} / {fused:.2f}", f"{ratio:.2f} [{low:.2f}, {high:.2f}]"))
        print("| " + " | ".join(cells) + " |")
    if args.plot:
        plot(results, args.plot)


if __name__ == "__main__":
    main()
