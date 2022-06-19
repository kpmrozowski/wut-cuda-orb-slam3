add_executable(tool_OpenCLTest main.cpp)
target_link_libraries(tool_OpenCLTest ${PROJECT_NAME}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
