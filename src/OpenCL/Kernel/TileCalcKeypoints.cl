#include "OpenCL/Kernel/key_point.hpp"

const sampler_t iSampler =  CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE |
                            CLK_FILTER_NEAREST;

#define PIXEL(idx) \
    read_imagei(image, iSampler, (int2)(idx/2+20, 20))

__kernel void tileCalcKeypoints_kernel(
    __read_only const image2d_t image,
    __global char* keypoints,   int kStep, int kOffset, int kRows, int kCols,
    __global int* descriptors, int dStep, int dOffset, int dRows, int dCols)
{
    const size_t id = get_global_id(0) / get_local_size(0);
    const size_t tid = get_global_id(0) % get_local_size(0);
    __global key_point_t *kpt = (__global key_point_t *)keypoints;
    descriptors[tid + id * get_local_size(0)] = (uchar)(0);
}
