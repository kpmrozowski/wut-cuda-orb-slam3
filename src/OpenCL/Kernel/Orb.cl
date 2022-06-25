#include "OpenCL/Kernel/key_point.hpp"

ushort2 make_ushort2(ushort x, ushort y)
{
    ushort2 res;
    res.x = x;
    res.y = y;
    return res;
}

bool is_rhs_greater(Point3b lhs, Point3b rhs) {
    if (lhs.x < rhs.x) {
        return true;
    } else if (lhs.x > rhs.x) {
        return false;
    }
    if (lhs.y < rhs.y) {
        return true;
    } else if (lhs.y > rhs.y) {
        return false;
    }
    if (lhs.z < rhs.z) {
        return true;
    }
    return false;
}

// filled with sample values, should be empty:
__global unsigned char c_pattern[sizeof(Point2i) * 512] = {3,4,5,6,0,1,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,7,0,1,2,3,4,5,6,7};
#define CV_PI 3.141592f

const sampler_t iSampler =  CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE |
                            CLK_FILTER_NEAREST;

#define GET_VALUE(idx) \
    read_imagei(image, iSampler, (float2)( \
        (float)loc.x + (float)(pattern[idx].x * cosA - pattern[idx].y * sinA)+0.5f, \
        (float)loc.y + (float)(pattern[idx].x * sinA + pattern[idx].y * cosA)+0.5f)).x

#define PIXEL(idx) \
    read_imagei(image, iSampler, (int2)(idx/2+20, 20))

__kernel void calcOrb_kernel(
    __read_only const image2d_t image,
    __global key_point_t* keypoints,   int kStep, int kOffset, int kRows, int kCols,
    // __global key_point_t* keypoints, // tak jest zle (CL_INVALID_KERNEL_ARGS)
    __global int* descriptors,   int dStep, int dOffset, int dRows, int dCols)
    // __global int* descriptors) // a w tym przypadku tak jest dobrze (albo albo)
{
    const size_t id = get_global_id(0) / get_local_size(0);
    const size_t tid = get_global_id(0) % get_local_size(0);
    ushort2 loc = make_ushort2(keypoints[id].pt.x, keypoints[id].pt.y);
    Point2i *pattern = (Point2i *)c_pattern + 16 * tid;

    const float factorPI = (float)(CV_PI/180.f);
    float angle = (float)keypoints[id].angle * factorPI;
    float cosA = cos(angle);
    float sinA = sin(angle);
    // __local int t[4];
    int t0, t1, val, idx;
    t0 = GET_VALUE(0);  t1 = GET_VALUE(1);  val  = (t0 < t1);
    t0 = GET_VALUE(2);  t1 = GET_VALUE(3);  val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4);  t1 = GET_VALUE(5);  val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6);  t1 = GET_VALUE(7);  val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8);  t1 = GET_VALUE(9);  val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10); t1 = GET_VALUE(11); val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12); t1 = GET_VALUE(13); val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14); t1 = GET_VALUE(15); val |= (t0 < t1) << 7;

    // TODO: zeros only (why?)
    // descriptors[tid + id * get_local_size(0)] = (uchar)(1000 * (t1 - t0));

    // debuging:
    // descriptors[tid + id * get_local_size(0)] = (uchar)(PIXEL(tid).x);
    descriptors[0  + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[1  + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[2  + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[3  + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[4  + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[5  + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[6  + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[7  + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[8  + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[9  + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[10 + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[11 + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[12 + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[13 + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[14 + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    descriptors[15 + id * get_local_size(0)] = (uchar)(keypoints[id].pt.x);
    // int4 test4;
    // test4.xyzw = (int4)(4, 5, 6, 7);
}
