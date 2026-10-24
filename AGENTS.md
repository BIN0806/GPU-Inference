# Project collaboration guidance

The user wants to understand and defend this project personally, and has authorized implementation in small, bloggable chunks. Build a coherent, reviewable increment with explanations and executed checks; keep the learning objective central. Do not turn a chunk into the entire inference system. For design-only or review-only requests, preserve that narrower scope.

Read README.md, the relevant ROADMAP.md phase, and recent docs/ENGINEERING_LOG.md entries before continuing implementation work. Distinguish recommendations and planned files from code that actually exists.

During each development session, update docs/ENGINEERING_LOG.md with design alternatives, decisions, changed files, problems, diagnostic evidence, solutions, tests/experiments, lessons, and next actions. Link code revisions and experiment IDs. Create or update a decision record for consequential architectural changes. Use docs/templates/EXPERIMENT.md for performance experiments. Do not reconstruct unsupported history or invent metrics.

Label work as proposed, implemented, tested, or measured. A test is passed only after execution. A speedup requires saved raw measurements with a clear baseline, environment, timing boundary, and correctness evidence. Preserve negative and inconclusive results.

Maintain docs/BUDGET.md when an infrastructure action or expense occurs. The total infrastructure limit is $100, including overhead, with an initial $20 checkpoint. This roadmap is not a request to provision GPU resources now. Any future provisioned resource needs cost accounting and verified cleanup within the user's authorized task.

Prefer local CPU work before paid GPU sessions. Keep a single-layer synthetic decoder distinct from real model inference and serving. Optimize only after correctness and a measurable hypothesis. Do not expand into a full serving framework by default.

Follow CONTRIBUTING.md for development conventions. When collaboration is requested, consult docs/TEAM_WORKFLOW.md for disjoint ownership, shared-interface changes, and integration. Project skills live in .agents/skills: kvlab-allocator, kvlab-cuda-kernels, kvlab-performance, kvlab-integration-review, and kvlab-engineering-record. Load only the skill relevant to the task. Their existence does not automatically call for subagents or override the user's mentoring preference.

For each implementation chunk, maintain a standalone post and its index under
`docs/blogs` locally. This entire directory, including its evidence files, is
gitignored and must never be staged, force-added, copied into tracked files, or
published as part of a code commit. Include the problem, alternatives,
invariants, relevant code, reproduction commands, executed checks, limitations,
and next learning step in the local post. Keep public implementation status,
test summaries, design decisions, and the engineering log in tracked docs;
do not create repository links to ignored files. Publishing the blog requires
a separate explicit user request that changes this local-only policy.
