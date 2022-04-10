file(GLOB CUDAFILES "*.cu" "*.cpp")
CUDA_ADD_LIBRARY(eden_resources ${CUDAFILES} STATIC)
target_compile_definitions(eden_resources PUBLIC "$<$<CONFIG:DEBUG>:DEBUG>")


set(CUDA_ARCH "-arch=sm_80" CACHE STRING "Value of the NVCC -arch option.")
if(${CMAKE_VERSION} VERSION_EQUAL 3.7 OR ${CMAKE_VERSION} VERSION_GREATER 3.7)
  option(AUTO_DETECT_ARCH "Detect local GPU compute arch automatically" ON)
endif()
message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
  ${CUDA_ARCH} 
  -std=c++17; 
  -ftemplate-backtrace-limit=0;
  --cudart;
  static;
  -O0 ; 
  --expt-relaxed-constexpr; 
  -fmad=true; 
  --expt-extended-lambda;
  --relocatable-device-code=false;
  -Xptxas 
  -dlcm=ca )

# target_compile_options(carcassonne_headless PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:my_cuda_option>) 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -allow-unsupported-compiler")

SET(CUDA_PROPAGATE_HOST_FLAGS ON)
SET(CUDA_SEPARABLE_COMPILATION OFF)
SET(CUDA_VERBOSE_BUILD OFF)
