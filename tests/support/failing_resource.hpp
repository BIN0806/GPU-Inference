#pragma once

#include <cstddef>
#include <memory_resource>
#include <new>

namespace kvlab::test {

// Isolated to a request's table allocations; does not intercept GoogleTest/global new.
class FailingResource final : public std::pmr::memory_resource {
 public:
  bool fail_next = false;
  std::size_t allocations = 0;
  std::size_t outstanding_bytes = 0;

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (fail_next) {
      fail_next = false;
      throw std::bad_alloc();
    }
    void* result = std::pmr::new_delete_resource()->allocate(bytes, alignment);
    ++allocations;
    outstanding_bytes += bytes;
    return result;
  }
  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    outstanding_bytes -= bytes;
    std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
};

}  // namespace kvlab::test
