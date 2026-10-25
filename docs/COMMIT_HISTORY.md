# Implementation chunks and Git history

The repository preserves independently buildable implementation chunks:

| Chunk | Commit subject | Previously executed Debug checks |
|---|---|---:|
| 01 | `feat(allocator): add generation-safe page pool (chunk 01)` | 12/12 |
| 02 | `feat(allocator): add sealed-page sharing and RAII leases (chunk 02)` | 32/32 |
| 03 | `feat(allocator): add transactional request tables (chunk 03)` | 52/52 |
| 04 | `feat(bench): add CPU workload replay and memory accounting (chunk 04)` | 67/67; affected CLI check repeated after timing refinement |
| 05 | `ci: verify sanitizer instrumentation and Linux CPU tests (chunk 05)` | 69/69 locally; UBSan 70/70; hosted results in the validation guide |

The history was initially reconstructed from validated delivery snapshots.
At the user's request, all three commits were subsequently rewritten to
exclude the entire local blog directory. Commit IDs changed. Implementation,
tests, CMake configuration, and dependency definitions remain byte-for-byte
identical to the corresponding original revisions. `.gitignore` and public
documentation reflect the local-only blog policy from the first commit.

## Inspect and reproduce a revision

```sh
git log --reverse --format=fuller
git worktree add --detach ../kvlab-history-check <commit-sha>
cd ../kvlab-history-check
cmake --preset debug
cmake --build --preset debug --parallel 4
ctest --preset debug
```

The first test-enabled configure fetches checksum-pinned GoogleTest unless
`-DKVLAB_GTEST_ARCHIVE=/absolute/path/to/archive.tar.gz` is supplied to CMake.
Preserve any work before removing a temporary checkout with `git worktree remove`.
See [dependencies](DEPENDENCIES.md) for offline and tests-disabled builds.

## Evidence and publication boundaries

The [engineering log](ENGINEERING_LOG.md) records executed configurations,
results, failure investigations, and limitations. Fresh isolated Debug builds
passed 12/32/52 checks during initial publication preparation. Those independent
builds ran concurrently; their durations are not performance measurements.
Release and UBSan results are the earlier recorded runs. Those initial history checks did not establish ASan or CI coverage. Current
CPU validation is recorded in [the validation guide](VALIDATION.md); chunk 04
adds CPU performance data. No GPU result is claimed.

Blog articles and their raw evidence remain only in the original local ignored
directory. They are absent from every commit in the rewritten published history
and are not included in a fresh clone. References to local-only records in
historical log entries do not promise a repository artifact. The history-only
cleanup did not rerun code tests; it verified unchanged code/test/build blobs
for every revision and checked all reachable trees for excluded content.

Future commits include code, tests, public design decisions, and engineering-log
updates. Write blog installments locally as a separate, ignored artifact. See
[contributor conventions](../CONTRIBUTING.md). Never merge pre-cleanup history
back into this repository; use a fresh clone when replacing an old checkout.

History rewriting removes branch references, but cannot guarantee deletion from
GitHub caches or third-party clones. See
[GitHub's history-removal documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

Chunk 04 adds the [benchmark contract](BENCHMARKS.md) and public CPU experiment
report. These are engineering records, maintained separately from the ignored
local blog. Its source and result provenance are linked from the report.
