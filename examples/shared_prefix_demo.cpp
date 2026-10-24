#include "kvlab/page_lease.hpp"

#include <iostream>

int main() {
  kvlab::PagePool pool(4, 16);
  auto prefix_a = kvlab::PageLease::try_allocate(pool);
  if (!prefix_a || pool.seal_full(*prefix_a->handle(), 16) != kvlab::SealResult::Sealed) {
    return 1;
  }
  // This is metadata-only: the 16 filled tokens are declared, not materialized KV data.
  auto prefix_b = prefix_a->try_share();
  auto tail_a = kvlab::PageLease::try_allocate(pool);
  auto tail_b = kvlab::PageLease::try_allocate(pool);
  if (!prefix_b || !tail_a || !tail_b) {
    return 1;
  }
  const auto prefix = prefix_b->handle();
  if (tail_a->handle()->id == tail_b->handle()->id || tail_a->try_share()) {
    return 1;
  }
  std::cout << "two logical prefixes, one physical page, references: "
            << pool.inspect(*prefix)->references << '\n';
  std::cout << "independent writable tails; allocated physical pages: "
            << pool.stats().allocated_pages << '\n';
  prefix_a->reset();
  tail_a->reset();
  if (pool.inspect(*prefix)->references != 1 || pool.stats().allocated_pages != 2) {
    return 1;
  }
  std::cout << "after owner A finishes, prefix references: "
            << pool.inspect(*prefix)->references << '\n';
  prefix_b->reset();
  tail_b->reset();
  pool.validate_invariants();
  std::cout << "after both finish, free pages: " << pool.stats().free_pages << '\n';
  return pool.stats().free_pages == 4 ? 0 : 1;
}
