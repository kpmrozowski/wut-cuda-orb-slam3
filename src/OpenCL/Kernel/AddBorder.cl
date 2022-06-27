#include "OpenCL/Kernel/key_point.hpp"

__kernel void addBorder_kernel(__global key_point_t *keypoints, int npoints, int minBorderX, int minBorderY, int octave, int size) {
    int tid = get_global_id(0);
    if (tid >= npoints) {
        return;
    }
    keypoints[tid].pt.x += minBorderX;
    keypoints[tid].pt.y += minBorderY;
    keypoints[tid].octave = octave;
    keypoints[tid].size   = size;
}
