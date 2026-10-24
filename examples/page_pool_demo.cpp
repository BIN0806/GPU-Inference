#include "kvlab/page_pool.hpp"

#include <iostream>
#include <vector>

int main() {
  kvlab::PagePool pool(4);
  std::vector<kvlab::PageHandle> handles;
  while (const auto handle = pool.try_allocate()) {
    handles.push_back(*handle);
    std::cout << "allocated page " << handle->id << ", generation " << handle->generation << '\n';
  }
  std::cout << "fifth allocation: exhausted\n";
  const auto old = handles.front();
  if (pool.release(old) != kvlab::ReleaseResult::Released) {
    return 1;
  }
  const auto replacement = pool.try_allocate();
  if (!replacement) {
    return 1;
  }
  handles.front() = *replacement;
  std::cout << "reused page " << replacement->id << ", generation " << replacement->generation << '\n';
  if (pool.release(old) != kvlab::ReleaseResult::StaleHandle) {
    return 1;
  }
  std::cout << "old handle: rejected\n";
  for (const auto handle : handles) {
    if (pool.release(handle) != kvlab::ReleaseResult::Released) {
      return 1;
    }
  }
  pool.validate_invariants();
  std::cout << "free pages after cleanup: " << pool.stats().free_pages << '\n';
  return pool.stats().free_pages == 4 ? 0 : 1;
}
