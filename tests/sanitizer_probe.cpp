// Intentional faults for validating the sanitizer runtime, never library code.
#include <iostream>
#include <limits>
#include <memory>
#include <string>

int main(int argc, char** argv) {
  std::cerr << "kvlab sanitizer probe entered main\n";
  if (argc != 2) {
    return 2;
  }
  const std::string mode(argv[1]);
  if (mode == "clean" || mode == "heap-overflow") {
    auto storage = std::make_unique<int[]>(4);
    volatile int* data = storage.get();
    // Volatile access keeps the fault observable in optimized test builds.
    const int index = mode == "clean" ? 0 : 4;
    data[index] = 42;
    std::cout << "probe value: " << data[index] << '\n';
    if (mode == "clean") {
      std::cout << "kvlab sanitizer clean probe passed\n";
    }
    return 0;
  }
  if (mode == "signed-overflow") {
    volatile int maximum = std::numeric_limits<int>::max();
    volatile int one = 1;
    volatile int result = maximum + one;
    std::cout << result << '\n';
    return 0;
  }
  return 2;
}
