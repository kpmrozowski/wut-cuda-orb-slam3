set(TMP_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CXX_COMPILER /usr/bin/hipcc)

add_library(eden_resources Ngpus_Ncpus.cpp)
target_link_libraries(eden_resources spdlog fmt amdhip64)

set(CMAKE_CXX_COMPILER ${TMP_COMPILER})
