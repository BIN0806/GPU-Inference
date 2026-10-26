cmake_minimum_required(VERSION 3.24)
project(kvlab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KVLAB_ENABLE_CUDA "Build CUDA components (not implemented yet)" OFF)
option(KVLAB_SANITIZE "Legacy alias: enable both host sanitizers" OFF)
option(KVLAB_ENABLE_ASAN "Enable host AddressSanitizer" OFF)
option(KVLAB_ENABLE_UBSAN "Enable host UndefinedBehaviorSanitizer" OFF)
set(KVLAB_GTEST_ARCHIVE "" CACHE FILEPATH "Optional local copy of the pinned GoogleTest archive")

if(KVLAB_ENABLE_CUDA)
  message(FATAL_ERROR "Current implementation is CPU-only. Configure with KVLAB_ENABLE_CUDA=OFF.")
endif()

add_library(kvlab_options INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(kvlab_options INTERFACE -Wall -Wextra -Wpedantic)
endif()
set(kvlab_sanitizers "")
if(KVLAB_ENABLE_ASAN OR KVLAB_SANITIZE)
  list(APPEND kvlab_sanitizers address)
endif()
if(KVLAB_ENABLE_UBSAN OR KVLAB_SANITIZE)
  list(APPEND kvlab_sanitizers undefined)
endif()
if(kvlab_sanitizers)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU" OR WIN32)
    message(FATAL_ERROR "Host sanitizers currently support Clang/GCC on macOS/Linux.")
  endif()
  list(JOIN kvlab_sanitizers "," kvlab_sanitizer_flags)
  target_compile_options(kvlab_options INTERFACE
    "-fsanitize=${kvlab_sanitizer_flags}" -fno-omit-frame-pointer)
  target_link_options(kvlab_options INTERFACE "-fsanitize=${kvlab_sanitizer_flags}")
endif()

add_library(kvlab_allocator STATIC src/allocator/page_pool.cpp src/allocator/page_lease.cpp
  src/allocator/request_state.cpp src/allocator/memory_accounting.cpp)
target_include_directories(kvlab_allocator PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(kvlab_allocator PUBLIC kvlab_options)

add_executable(page_pool_demo examples/page_pool_demo.cpp)
target_link_libraries(page_pool_demo PRIVATE kvlab_allocator)

add_executable(shared_prefix_demo examples/shared_prefix_demo.cpp)
target_link_libraries(shared_prefix_demo PRIVATE kvlab_allocator)

add_executable(request_state_demo examples/request_state_demo.cpp)
target_link_libraries(request_state_demo PRIVATE kvlab_allocator)

add_library(kvlab_workload STATIC src/bench/workload.cpp)
target_link_libraries(kvlab_workload PUBLIC kvlab_allocator)
add_executable(allocator_bench bench/allocator_bench.cpp)
target_link_libraries(allocator_bench PRIVATE kvlab_workload)
target_compile_definitions(allocator_bench PRIVATE KVLAB_BUILD_TYPE="$<CONFIG>")

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  set(kvlab_gtest_url "https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.17.0")
  if(KVLAB_GTEST_ARCHIVE)
    set(kvlab_gtest_url "${KVLAB_GTEST_ARCHIVE}")
  endif()
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL "${kvlab_gtest_url}"
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  )
  FetchContent_MakeAvailable(googletest)
  add_executable(page_pool_test tests/page_pool_test.cpp tests/page_sharing_test.cpp
    tests/page_lease_test.cpp tests/request_state_test.cpp tests/request_model_test.cpp)
  target_link_libraries(page_pool_test PRIVATE kvlab_allocator GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(page_pool_test DISCOVERY_MODE PRE_TEST)
  add_executable(workload_test tests/memory_accounting_test.cpp tests/workload_test.cpp)
  target_link_libraries(workload_test PRIVATE kvlab_workload GTest::gtest_main)
  gtest_discover_tests(workload_test DISCOVERY_MODE PRE_TEST)
  # Test tooling is required so an absent interpreter cannot silently remove checks.
  find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
  add_test(NAME BenchmarkOutputContract COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_output_test.py" "$<TARGET_FILE:allocator_bench>")
  add_test(NAME ValidationTools COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/validation_tools_test.py")
  add_test(NAME RmsNormOracle COMMAND "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python" "${Python3_EXECUTABLE}"
    -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/tests/python" -p test_rmsnorm_oracle.py)
  add_test(NAME GpuPreflightContracts COMMAND "${Python3_EXECUTABLE}"
    -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/tests/python" -p test_gpu_preflight.py)
  set_tests_properties(RmsNormOracle GpuPreflightContracts PROPERTIES TIMEOUT 30)
  set(kvlab_expected_sanitizers "none")
  if(kvlab_sanitizers)
    set(kvlab_expected_sanitizers "${kvlab_sanitizer_flags}")
    add_executable(sanitizer_probe tests/sanitizer_probe.cpp)
    target_link_libraries(sanitizer_probe PRIVATE kvlab_options)
    add_test(NAME SanitizerRuntime COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/sanitizer_runtime_test.py"
      "$<TARGET_FILE:sanitizer_probe>" --sanitizers "${kvlab_sanitizer_flags}"
      --log-dir "${CMAKE_CURRENT_BINARY_DIR}/sanitizer-logs")
    set_tests_properties(SanitizerRuntime PROPERTIES TIMEOUT 70)
  endif()
  add_test(NAME SanitizerCompileFlags COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_sanitizer_flags.py"
    --source-dir "${CMAKE_CURRENT_SOURCE_DIR}" --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
    --expected "${kvlab_expected_sanitizers}")
  add_test(NAME PagePoolDemo COMMAND page_pool_demo)
  add_test(NAME SharedPrefixDemo COMMAND shared_prefix_demo)
  add_test(NAME RequestStateDemo COMMAND request_state_demo)
endif()
