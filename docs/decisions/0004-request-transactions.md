# ADR 0004: reservation, commitment, and transactional request ownership

Date: 2026-10-23. Status: implemented; all 52 checks passed in Debug, Release, and UBSan-only builds. ASan remains open.

The growth path below describes the original chunk 03 baseline.
[ADR 0007](0007-request-table-growth.md) updates growth and retained-capacity
semantics in chunk 06; sharing and move assignment still follow this record.

## Contract

A move-only `RequestState` borrows one PagePool and owns an ordered table of PageLease entries. Page IDs within a request are distinct. `reserved_length` bounds planned capacity; `length` counts committed tokens. The table contains exactly ceil(reserved_length / tokens_per_page) entries, possibly including uncommitted pages. Only fully committed pages are sealed. Any remaining partial tail or reserved page is uniquely owned and writable.

`reserve_for_length(n)` is monotonic. Smaller/equal targets do nothing; successful larger targets reserve pages but do not change committed length or seals. `commit_length(n)` rejects shrinking and exceeding the reservation, validates every newly touched page before changing metadata, then seals newly completed pages and publishes length. It is a caller declaration that tokens have been materialized, not a payload write, validation of KV data, or GPU fence. Token lookup exposes only committed positions; page-handle inspection includes reserved pages for future writes.

`share_prefix_from(source,k)` requires an empty destination, the identical pool, and at least k complete committed source pages. It retains exactly those pages transactionally and sets both destination lengths to k times page size. It does not copy a partial source tail; callers append independently. Automatic prefix search, payload equality checks, copy-on-write, and multi-layer caches remain outside scope.

## Transaction and failure guarantees

Construct a temporary candidate table with empty leases, allocating its host storage before acquiring pages/references. For growth, acquire all new pages in their final logical positions while old owners remain in the original table. Only after success, move old leases into the candidate and swap tables. Lease moves and the final swap cannot throw under the stated resource contract. For sharing, acquire the prefix's references in the candidate and swap only after all retains succeed.

Pool exhaustion and invalid operations return typed statuses. Host allocation and generation exhaustion exceptions propagate. Candidate destruction releases any partially acquired pages/references. Existing request handles, committed/reserved lengths, and owner counts remain unchanged on ordinary failure. Free capacity is restored after a partial acquisition fails, but generations of briefly acquired pages and free-list order need not be restored. Rolling generations backward could make old handles valid and is explicitly excluded.

Commit preflights all newly touched pages before sealing any. Under the single-thread, exclusive metadata-writer contract, the subsequent seal calls cannot fail. An unexpected violation terminates, as with PageLease's invalid release. Borrowed request handles must not be manually released, retained, or sealed by callers. No recovery guarantee covers arbitrary external corruption or destruction of the borrowed pool/resource.

## Metadata allocation and move semantics

Use C++17 `std::pmr::vector<PageLease>` with a borrowed memory resource, defaulting to the standard default resource. A test resource can fail a chosen allocation without intercepting global new or affecting GoogleTest. This adds no third-party dependency. Reject null resources; resources must outlive all objects using their storage.

A candidate always uses the destination table's resource, so its swap uses equal allocators. Resource-sensitive container operations require this care; see the [C++ working draft allocator-aware container requirements](https://eel.is/c++draft/container.reqmts). Move construction transfers storage/resource association without allocating and detaches the source. Move assignment keeps the destination resource: allocate a candidate first, move source entries only after allocation succeeds, swap, then release the old destination's leases. Consequently move assignment can throw; it is intentionally not noexcept.

## Alternatives and cost

- **Grow the live table in place:** may be faster, but needs careful rollback of partially acquired entries and distinguishes visible state from hidden capacity changes. A full candidate table gives this first version a simple commit point.
- **Reserve maximum request metadata up front:** removes repeated table allocation but uses more host memory and requires a maximum-length policy. Revisit during workload measurement.
- **One operation that allocates and commits:** easy for a CPU simulation but would hide the future gap between device writes and valid token publication. Separate operations now preserve that boundary.
- **Implicitly copy request objects:** would hide retain operations and mutable-tail policy. Requests remain move-only; prefix sharing is explicit and rejects a nonempty destination.
- **Swap different PMR allocators or mark all moves noexcept:** ignores resource lifetime/equality or allocation failure. The explicit assignment transaction preserves ownership and makes the real exception contract visible.

Crossing a page boundary currently allocates and initializes O(total target pages) temporary metadata and moves old owners in O(old pages). Repeated page-at-a-time growth can therefore produce quadratic total metadata work. Growth within an existing last page changes only a scalar. Prefix sharing is O(prefix pages); reset releases O(owned pages) and may retain host vector capacity until later replacement/destruction. These are algorithmic properties, not measured performance results.

## Validation and next gate

See chunk 03 evidence (local-only record). Test page boundaries, invisible reservations, committed address translation, OOM retry, host bad_alloc, failure after partial acquisition, failure after partial retain, extreme size_t arithmetic, moves across resources/pools, cancellation, and a multi-request independent model. Shared test hooks are defined in one header to avoid inconsistent definitions across test translation units.

Next measure synthetic allocator/request workloads and fragmentation before changing the transaction design. ASan remains an open environment check; no GPU rental is justified by this increment alone.
