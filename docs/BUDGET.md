# Infrastructure budget and session ledger

Updated 2026-10-27. **Total hard ceiling: $100. Initial checkpoint: $20 all-in. Actual project spend recorded so far: $0. No rental is active from this work.** The plan is not an instruction to provision now.

## Envelopes

| Category | Maximum |
|---|---:|
| Core GPU compute, including setup/idle/compile | $60 |
| Storage | $5 |
| Transfers and other infrastructure | $5 |
| Tax/payment reserve | $10 |
| Contingency | $15 |
| Optional extension | $5 |
| Total | $100 |

The first $20 is contained within these envelopes: $14 compute, $2 storage/transfers, $2 tax, $2 buffer. Core compute provisional split: $2 preflight, $12 RMSNorm, $26 paged attention, $16 systems measurements, $4 reproduction. These are plans, not bills.

Cumulative review targets: $20 after normalization evidence; $45 after correct paged decode; $70 after integrated experiments; $85 after final reproduction. Keep the final $15 protected until outstanding costs are reconciled. Spending less is a success.

## Ledger

| Session | Date | Provider/resource | New cash outlay | Accrued usage | Pending liability | Remaining prepaid credit | Cleanup evidence | Result |
|---|---|---|---:|---:|---:|---:|---|---|
| Planning | 2026-10-23 | None | $0 | $0 | $0 | N/A | No resource provisioned | Roadmap only |
| Chunk 01 | 2026-10-23 | Local CPU only | $0 | $0 | $0 | N/A | No cloud resource provisioned | Page-pool implementation and blog |
| Chunk 02 | 2026-10-23 | Local CPU only | $0 | $0 | $0 | N/A | No cloud resource provisioned | Reference-counted sharing, leases, tests, and blog |
| Chunk 03 | 2026-10-23 | Local CPU only | $0 | $0 | $0 | N/A | No cloud resource provisioned | Request transactions, failure tests, and blog |
| Chunk 04 | 2026-10-24 | Local CPU only | $0 | $0 | $0 | N/A | No remote compute/storage provisioned by this session | Accounting tests, two retained 108-run sweeps, and local plots |
| Chunk 05 | 2026-10-25 | Local CPU and standard public-repository GitHub Actions | $0 | $0 | $0 | N/A | All 5 jobs in run 40172396157 completed; artifact count 0; no GPU or persistent volume | Sanitizer probes and Linux validation |
| Chunk 06 | 2026-10-25 | Local CPU and standard public Linux CI | $0 | $0 | $0 | N/A | 108 local processes and all 5 jobs in run 40232916754 completed; artifact count 0; no GPU | Request growth tests and paired host-memory/time comparison |
| Chunk 07 | 2026-10-25 | Local CPU and standard public Linux CI | $0 | $0 | $0 | N/A | All 6 jobs in run 40270856228 completed; artifact count 0; no GPU or persistent volume | 218 reference cases locally and Linux, C++/sanitizer tests, and local tool inventory |
| Chunk 08 | 2026-10-26 | Local CPU and standard public Linux CI | $0 | $0 | $0 | N/A | All 7 jobs in run 40427805616 completed; artifact uploads 0; no GPU or persistent volume | Native CUDA compiled/linked, host checks passed; GPU execution untested |

Track cash purchases and resource consumption separately. Using prepaid credit is not a second cash outlay. Include mandatory deposits/fees when checking whether total cash commitment exceeds the ceiling. Do not rely on future refunds, free credits, or automatic account shutdown to enforce the limit.

## Before each session

- Experiment ID, hypothesis, exact command, expected maximum runtime, artifact destination.
- Quoted GPU count/model/rate, selected disk quantity and hourly component, bandwidth prices, tax/fees, minimum funding amount, interruption policy.
- All already-running or retained resources and their projected cost until cleanup.
- Projected total including protected reserve; fit within released tranche and $100.
- Intended stop time and provider cleanup method. Disable automatic top-up. Do not enter secrets in this file.

## After each session

- Export code diff, raw data, configs, environment, and relevant logs; verify local copies.
- Stop/delete the intended compute resource through the provider and verify state. An SSH disconnect or Linux shutdown is not billing evidence.
- Delete unneeded volumes only after verifying exports; confirm what storage remains billable.
- Reconcile invoice/dashboard amounts, cash outlay, usage, credit, and pending charges.
- Record what the spend established. Release more budget only for an evidence-backed next experiment.

Rates and availability are transient. See [research notes](RESEARCH_NOTES.md) and [roadmap section 8](../ROADMAP.md#8-gpu-cost-analysis-and-spending-strategy). Refresh the deployment quote before each purchase.

## CPU CI cost boundary (verified 2026-10-25)

The repository was verified public before enabling CI. GitHub states that
standard hosted runners are free for public repositories; workflow logs and job
summaries do not count against artifact storage. This workflow uses only the
standard Ubuntu runner, with no Actions cache or artifact uploads. It has a
10-minute per-job timeout, at most two concurrent matrix jobs plus independent CPU reference and CUDA compile jobs,
and cancels older runs on the same ref. A job guard skips execution if the repository is private.
Larger runners, GPU services, paid storage, and a future visibility change need
a new cost review. [GitHub Actions billing documentation](https://docs.github.com/en/billing/concepts/product-billing/github-actions).

Free CI does not produce GPU performance evidence and does not release the
first $20 rental checkpoint. Record completed hosted runs in the validation guide;
do not describe merely queued work as completed cleanup.


## Chunk 09 preparation

Native session orchestration and cancellation tests ran locally at $0. The user
confirmed no GPU host is available and requested workflow preparation. No
rental, credit purchase, retained cloud volume, or paid profiling was initiated.
Hosted CPU validation is pending at the implementation commit; record its
completed state after the jobs finish. The first $20 checkpoint remains unopened.
