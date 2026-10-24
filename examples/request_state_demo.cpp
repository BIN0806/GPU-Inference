#include "kvlab/request_state.hpp"

#include <iostream>

int main() {
  kvlab::PagePool pool(5, 4);
  kvlab::RequestState first(pool), second(pool);
  if (first.reserve_for_length(9) != kvlab::RequestStatus::Ok || first.length() != 0) {
    return 1;
  }
  std::cout << "reserved 9 token slots, committed tokens: " << first.length() << '\n';
  // Metadata simulation: the caller declares that these KV positions are now ready.
  if (first.commit_length(9) != kvlab::RequestStatus::Ok ||
      second.share_prefix_from(first, 2) != kvlab::RequestStatus::Ok ||
      second.reserve_for_length(9) != kvlab::RequestStatus::Ok ||
      second.commit_length(9) != kvlab::RequestStatus::Ok) {
    return 1;
  }
  if (first.page_handle(2)->id == second.page_handle(2)->id) {
    return 1;
  }
  std::cout << "two requests share 8 prefix tokens, with independent tails; physical pages: "
            << pool.stats().allocated_pages << '\n';
  // Two extra pages are needed, but only one is free.
  if (second.reserve_for_length(17) != kvlab::RequestStatus::PoolExhausted ||
      second.length() != 9 || second.reserved_length() != 9 || second.page_count() != 3) {
    return 1;
  }
  std::cout << "failed growth preserved committed/reserved lengths: " << second.length()
            << '/' << second.reserved_length() << '\n';
  first.reset();
  if (second.reserve_for_length(17) != kvlab::RequestStatus::Ok ||
      second.commit_length(17) != kvlab::RequestStatus::Ok) {
    return 1;
  }
  second.validate_invariants();
  second.reset();
  pool.validate_invariants();
  std::cout << "after completion, free pages: " << pool.stats().free_pages << '\n';
  return pool.stats().free_pages == 5 ? 0 : 1;
}
