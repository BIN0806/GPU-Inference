"""Plot saved run metrics; optional Matplotlib dependency, no benchmark execution."""
import argparse
import csv
from pathlib import Path
import statistics


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with (args.directory / "runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    pages = sorted({int(row["page_tokens"]) for row in rows})
    sharing = sorted({int(row["sharing"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    plots = [("admission_rejection_fraction", "Rejected admissions / attempted admissions"),
             ("sampled_mean_tail_slack_tokens", "Mean sampled tail slack (physical token slots)"),
             ("aggregate_ns_per_offered_event", "Whole-trace time / offered events (ns/event)"),
             ("metadata_allocation_calls", "Request-table allocation calls per trace")]
    for ax, (metric, label) in zip(axes.flat, plots):
        for share in sharing:
            groups = [[float(row[metric]) for row in rows if int(row["page_tokens"]) == page
                       and int(row["sharing"]) == share and row[metric] != ""] for page in pages]
            x = [page for page, values in zip(pages, groups) if values]
            valid = [values for values in groups if values]
            medians = [statistics.median(values) for values in valid]
            ax.errorbar(x, medians,
                        yerr=[[m - min(v) for m, v in zip(medians, valid)],
                              [max(v) - m for m, v in zip(medians, valid)]],
                        marker="o", capsize=3, label=f"{share}% offered sharing")
        ax.set(xlabel="Tokens per page", ylabel=label, xticks=pages)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("CPU metadata replay at fixed token-slot budget\nMedians and observed min–max across seeds/repeats; not confidence intervals")
    fig.savefig(args.directory / "allocator-tradeoffs.png", dpi=160)
    fig.savefig(args.directory / "allocator-tradeoffs.svg")
    plt.close(fig)

    chosen = next(row for row in rows if int(row["page_tokens"]) == pages[0] and int(row["sharing"]) == sharing[0])
    with (args.directory / chosen["run"] / "events.csv").open() as stream:
        events = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for allocated in (False, True):
        values = sorted(int(row["elapsed_ns"]) for row in events
                        if row["outcome"] == "appended" and (int(row["metadata_allocations"]) > 0) == allocated)
        if values:
            ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post",
                    label=f"{'with allocation' if allocated else 'within page'} (n={len(values)})")
    ax.set(xlabel="Observed CPU append duration (ns, timer included)", ylabel="Empirical cumulative fraction",
           xscale="symlog", title=f"One run: {chosen['run']}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(args.directory / "append-latency-cdf.png", dpi=160)
    fig.savefig(args.directory / "append-latency-cdf.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
