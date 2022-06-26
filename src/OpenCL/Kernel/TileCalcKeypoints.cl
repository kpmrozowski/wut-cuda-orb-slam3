#include "OpenCL/Kernel/key_point.hpp"

const sampler_t iSampler =  CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE |
                            CLK_FILTER_NEAREST;

#define PIXEL(imgX, imgY) \
    read_imagei(img, iSampler, (int2)(imgX, imgY))

bool isKeyPoint2(
    const image2d_t img,
    const int i,
    const int j,
    const int threshold,
    __global int* scoreMat,
    unsigned short scoreMatCols)
{
    int v;
    uint C[4] = {0,0,0,0};

    C[2] |= (uint)(PIXEL(i - 3, j - 1).x) << 8;
    C[2] |= (uint)(PIXEL(i - 3, j).x);
    C[1] |= (uint)(PIXEL(i - 3, j + 1).x) << (3 * 8);

    C[2] |= (uint)(PIXEL(i - 2, j - 2).x) << (2 * 8);
    C[1] |= (uint)(PIXEL(i - 2, j + 2).x) << (2 * 8);

    C[2] |= (uint)(PIXEL(i - 1, j - 3).x) << (3 * 8);
    C[1] |= (uint)(PIXEL(i - 1, j + 3).x) << 8;

    C[3] |= (uint)(PIXEL(i, j - 3).x);
    v     = (int)(PIXEL(i, j).x);
    C[1] |= (uint)(PIXEL(i, j + 3).x);

    // int d1 = diffType(v, C[1] & 0xff, threshold);
    // int d2 = diffType(v, C[3] & 0xff, threshold);

    // if ((d1 | d2) == 0) {
    //   return false;
    // }
    C[3] |= (uint)(PIXEL(i + 1, j - 3).x) << 8;
    C[0] |= (uint)(PIXEL(i + 1, j + 3).x) << (3 * 8);

    C[3] |= (uint)(PIXEL(i + 2, j - 2).x) << (2 * 8);
    C[0] |= (uint)(PIXEL(i + 2, j + 2).x) << (2 * 8);

    C[3] |= (uint)(PIXEL(i + 3, j - 1).x) << (3 * 8);
    C[0] |= (uint)(PIXEL(i + 3, j).x);
    C[0] |= (uint)(PIXEL(i + 3, j + 1).x) << 8;

    int mask1 = 0;
    int mask2 = 0;

    // calcMask(C, v, threshold, mask1, mask2);
    // if (isKeyPoint(mask1, mask2)) {
    //     scoreMat[i + j * scoreMatCols] = cornerScore(C, v, threshold);
    //     return true;
    // } else {
    //     scoreMat[i + j * scoreMatCols] = 0;
    //     return false;
    // }
    return false;
}

__kernel void tileCalcKeypoints_kernel(
    __private unsigned short gridDimX,
    __private unsigned short blockDimX,
    __private unsigned short blockDimY,
    __read_only const image2d_t img,
    __private int imgRows,
    __private int imgCols,
    __global short2* kpLoc,   int klStep, int klOffset, int klRows, int klCols,
    __global float* kpScore, int ksStep, int ksOffset, int ksRows, int ksCols,
    __private unsigned int maxKeypoints,
    __private unsigned int highThreshold,
    __private unsigned int lowThreshold,
    __global int* scoreMat, int sStep, int sOffset, int sRows, int sCols,
    __private unsigned short scoreMatCols,
    __global unsigned int* counterPtr
    )
{
    const size_t workGroupId = get_global_id(0) / get_local_size(0);
    const size_t threadId = get_global_id(0) % get_local_size(0);
    struct         { unsigned short x; unsigned short y; }
        blockDim = { blockDimX,        blockDimY };
    struct         { unsigned short x;       unsigned short y; }
        blockIdx = { workGroupId % gridDimX, workGroupId / gridDimX };
    struct          { unsigned short x;     unsigned short y; }
        threadIdx = { threadId % blockDimX, threadId / blockDimX };
    // unsigned short j = threadIdx.x + blockIdx.x * blockDim.x + 3;
    // unsigned short i = (threadIdx.y );
    const int j = threadIdx.x + blockIdx.x * blockDim.x + 3;
    const int i = (threadIdx.y + blockIdx.y * blockDim.y) * 4 + 3;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    __local bool hasKp;
    if (tid == 0) {
      hasKp = false;
    }

    bool isKp[4] = {0};
    for (int t = 0; t < 4; ++t) {
      if (i+t < imgRows - 3 && j < imgCols - 3) {
        isKp[t] = isKeyPoint2(img, i+t, j, highThreshold, scoreMat, scoreMatCols);
      }
    }
}
