cmake_minimum_required(VERSION 3.14)
project(ORB_SLAM3_tests)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    ORB_SLAM3_tests
    cv_ocl_tests.cpp
)
target_link_libraries(
    ORB_SLAM3_tests
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(ORB_SLAM3_tests)
