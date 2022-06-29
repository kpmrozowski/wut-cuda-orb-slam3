#include "OpenCL/Kernel/key_point.hpp"

__kernel void addBorder_kernel(
    __global uchar* keypoints_, int kStep, int kOffset, int kRows, int kCols,
    uint npoints,
    float minBorderX,
    float minBorderY,
    float octave,
    float size)
{
    int tid = get_global_id(0);
    if (tid >= npoints) {
        return;
    }
    __global key_point_t *keypoints = (__global key_point_t *)keypoints_;
    keypoints[tid].pt.x += minBorderX;
    keypoints[tid].pt.y += minBorderY;
    keypoints[tid].octave = octave;
    keypoints[tid].size   = size;
}
