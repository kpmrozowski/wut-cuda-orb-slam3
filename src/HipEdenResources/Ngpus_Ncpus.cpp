#include <Eden_resources/Ngpus_Ncpus.h>
#include <thread>
#include <spdlog/spdlog.h>
#include <hip/hip_runtime.h>

unsigned Eden_resources::get_gpus_count() {
    int count = 0;
    unsigned cuda_count = 0;
    
    hipGetDeviceCount(&count);
    if(count == 0) {
        spdlog::error("There is no device.");
        return cuda_count;
    }
    for(int i = 0; i < count; ++i) {
        hipDeviceProp_t prop;
        if(hipGetDeviceProperties(&prop, i) == hipSuccess) {
            if (prop.major >= 1) {
                ++cuda_count;
            }
            spdlog::info("[{}] --{}", i, prop.name);
        }
    }
    if(cuda_count == 0) {
        spdlog::error("There is no device supporting CUDA.");
    }
    return cuda_count;
}

unsigned Eden_resources::get_cpus_count() {
    const unsigned processor_count = std::thread::hardware_concurrency();
    if(processor_count == 0) {
        spdlog::error("Found 0 cpus");
    }
    return processor_count;
}
