#include "OpenCL/Kernel/key_point.hpp"

const sampler_t iSampler =  CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE |
                            CLK_FILTER_NEAREST;

#define PIXEL(idx) \
    read_imagei(image, iSampler, (int2)(idx/2+20, 20))

__kernel void tileCalcKeypoints_kernel(
    __read_only const image2d_t image,
    __global short2* kpLoc,   int klStep, int klOffset, int klRows, int klCols,
    __global float* kpScore, int ksStep, int ksOffset, int ksRows, int ksCols,
    unsigned int maxKeypoints,
    unsigned int highThreshold,
    unsigned int lowThreshold,
    __global int* scoreMat, int sStep, int sOffset, int sRows, int sCols,
    __global unsigned int* counter_ptr
    )
{
    // const size_t id = get_global_id(0) / get_local_size(0);
    // const size_t tid = get_global_id(0) % get_local_size(0);
    // __global key_point_t *kpt = (__global key_point_t *)keypoints;
    // descriptors[tid + id * get_local_size(0)] = (uchar)(0);
}
