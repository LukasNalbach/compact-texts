cmake_minimum_required(VERSION 3.21)

# texts stored with one byte, ceil(log2 sigma) bits or a variable number of bits per symbol, with random access and
# word-wise comparisons
project(compact-texts LANGUAGES C CXX)

# add external/string-utils, if the parent project has not added it
if(NOT TARGET string_utils)
  file(GLOB compact_texts_string_utils_contents "${CMAKE_CURRENT_SOURCE_DIR}/external/string-utils/*")

  if(compact_texts_string_utils_contents STREQUAL "")
    find_package(Git REQUIRED)
    message(STATUS "Downloading the git submodule external/string-utils")
    execute_process(COMMAND "${GIT_EXECUTABLE}" submodule update --init -- external/string-utils
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  # the prefix of the options and environment variables (COMPACT_TEXTS_MARCH_NATIVE, COMPACT_TEXTS_TEST_SECONDS, ...)
  if(PROJECT_IS_TOP_LEVEL)
    set(STRING_UTILS_PREFIX COMPACT_TEXTS)
    include("${CMAKE_CURRENT_SOURCE_DIR}/external/string-utils/cmake/settings.cmake")
  endif()

  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/string-utils")
endif()

option(COMPACT_TEXTS_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})

add_library(compact_texts INTERFACE)
target_include_directories(compact_texts INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(compact_texts INTERFACE string_utils OpenMP::OpenMP_CXX)

if(COMPACT_TEXTS_BUILD_TESTS)
  include(GoogleTest)
  enable_testing()

  string_utils_add_executable(test-compact-texts tests/test-compact-texts.cpp "${CMAKE_BINARY_DIR}/tests/"
    compact_texts string_utils_test)

  gtest_discover_tests(test-compact-texts)
endif()
