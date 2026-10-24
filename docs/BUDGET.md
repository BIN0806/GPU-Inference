# Infrastructure budget and session ledger

Updated 2026-10-23. **Total hard ceiling: $100. Initial checkpoint: $20 all-in. Actual project spend recorded so far: $0. No rental is active from this work.** The plan is not an instruction to provision now.

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
