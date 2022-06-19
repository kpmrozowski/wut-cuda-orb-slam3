# include(FetchContent)
# FetchContent_Declare(
#         googletest
#         URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
# )
# For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
        ORB_SLAM3_tests
        OpenCLTest.cpp
)
target_link_libraries(
        ORB_SLAM3_tests
        ORB_SLAM3
        gtest
        gtest_main
        OpenCL
)

target_include_directories(ORB_SLAM3_tests PRIVATE "/usr/include/opencv4")

include(GoogleTest)
gtest_discover_tests(ORB_SLAM3_tests)
