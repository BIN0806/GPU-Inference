---
name: kvlab-engineering-record
description: "Maintain this project's contemporaneous engineering log, design decisions, experiment provenance, and cost record from actual changes and evidence. Use for session documentation or technical writeups."
---

# Engineering evidence record

Read the relevant [log entries](../../../docs/ENGINEERING_LOG.md), code diff, executed command/results, experiment artifacts, and [budget ledger](../../../docs/BUDGET.md). Use [the existing template](../../../docs/templates/EXPERIMENT.md) for experiments. Avoid reconstructing unsupported history from the final code alone.

For a session entry, identify the question, prediction, alternatives/tradeoffs, decision, changed files/functions, actual failure if any, diagnostic evidence, solution, executed verification, limitations, learning, and next action. Link a short meaningful snippet/diff and raw data where available. Mark missing evidence explicitly; proposed commands are not executed tests.

Separate proposed, implemented, tested, and measured states. An anticipated risk is not a bug encountered; an error fixed in prose is not a tested code fix; a provider specification is not measured performance. Before/after comparisons need both revisions, equivalent conditions, units, and uncertainty. Preserve negative/inconclusive outcomes and disconfirming evidence.

Create an ADR for consequential scope, layout, precision, lifetime, scheduling, or dependency choices. Capture context, alternatives, decision, implications, and a condition for revisiting it. Do not create an ADR for routine naming or formatting. Update status when implementation/tests establish new evidence rather than silently implying the original recommendation was validated.

Keep chronology and provenance: timestamp in the user's timezone, use stable experiment/run IDs, append corrections to earlier evidence, and link model/config/toolchain revisions. Summarize large logs; keep their source paths/checksums. Protect credentials and unrelated personal/account data from publication.

For infrastructure, distinguish prepaid cash outlay from consumption and outstanding charges; do not count the same purchase twice. Record actual resource cleanup only with observed evidence. An intention to delete later is not a completed cleanup entry.

If another worker owns shared documentation, return a proposed entry and references to the integrator instead of racing to append. For an authorized documentation edit, update only the relevant records and verify links/status consistency. For portfolio prose, preserve the single-layer versus real-model boundary and fill numerical achievements only from actual experiments.

Return what records changed, the source evidence, unresolved gaps, and any next documentation action. This role does not generate missing results, perform speculative GPU experiments, or send/publish a writeup without task authorization.
