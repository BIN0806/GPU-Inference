cmake_minimum_required(VERSION 3.24)
project(kvlab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KVLAB_ENABLE_CUDA "Build CUDA components (not implemented in chunk 01)" OFF)
option(KVLAB_SANITIZE "Enable host AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
set(KVLAB_GTEST_ARCHIVE "" CACHE FILEPATH "Optional local copy of the pinned GoogleTest archive")

if(KVLAB_ENABLE_CUDA)
  message(FATAL_ERROR "Chunk 01 is CPU-only. Configure with KVLAB_ENABLE_CUDA=OFF.")
endif()

add_library(kvlab_options INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(kvlab_options INTERFACE -Wall -Wextra -Wpedantic)
endif()
if(KVLAB_SANITIZE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU" OR WIN32)
    message(FATAL_ERROR "KVLAB_SANITIZE currently supports Clang/GCC on macOS/Linux.")
  endif()
  target_compile_options(kvlab_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(kvlab_options INTERFACE -fsanitize=address,undefined)
endif()

add_library(kvlab_allocator STATIC src/allocator/page_pool.cpp)
target_include_directories(kvlab_allocator PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(kvlab_allocator PUBLIC kvlab_options)

add_executable(page_pool_demo examples/page_pool_demo.cpp)
target_link_libraries(page_pool_demo PRIVATE kvlab_allocator)

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
  add_executable(page_pool_test tests/page_pool_test.cpp)
  target_link_libraries(page_pool_test PRIVATE kvlab_allocator GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(page_pool_test DISCOVERY_MODE PRE_TEST)
  add_test(NAME PagePoolDemo COMMAND page_pool_demo)
endif()
