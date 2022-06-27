add_library(orb_benchmark include/orb/Benchmark.h src/Benchmark.cpp)
target_include_directories(orb_benchmark PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_subdirectory(Example)