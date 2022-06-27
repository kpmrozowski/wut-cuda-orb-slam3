#include "OpenCL/Kernel/c_table.cl"

const sampler_t iSampler =  CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE |
                            CLK_FILTER_NEAREST;

const sampler_t fSampler =  CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE |
                            CLK_FILTER_LINEAR;

#define PIXEL(imgY, imgX) \
    read_imagei(img, fSampler, (float2)(imgX + 0.5f, imgY + 0.5f))

#define PIXEL_I(imgY, imgX) \
    read_imagei(img, iSampler, (int2)(imgX, imgY))

inline int diffType(const int v, const int x, const int th)
{
    const int diff = x - v;
    return (int)(diff < -th) + ((int)(diff > th) << 1);
}

void calcMask(const uint* C, const int v, const int th, int* mask1_, int* mask2_)
{
    int mask1 = 0;
    int mask2 = 0;

    int d1, d2;

    d1 = diffType(v, C[0] & 0xff, th);
    d2 = diffType(v, C[2] & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 0;
    mask2 |= ((d1 & 2) >> 1) << 0;

    mask1 |= (d2 & 1) << 8;
    mask2 |= ((d2 & 2) >> 1) << 8;



    d1 = diffType(v, C[1] & 0xff, th);
    d2 = diffType(v, C[3] & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 4;
    mask2 |= ((d1 & 2) >> 1) << 4;

    mask1 |= (d2 & 1) << 12;
    mask2 |= ((d2 & 2) >> 1) << 12;



    d1 = diffType(v, (C[0] >> (2 * 8)) & 0xff, th);
    d2 = diffType(v, (C[2] >> (2 * 8)) & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 2;
    mask2 |= ((d1 & 2) >> 1) << 2;

    mask1 |= (d2 & 1) << 10;
    mask2 |= ((d2 & 2) >> 1) << 10;



    d1 = diffType(v, (C[1] >> (2 * 8)) & 0xff, th);
    d2 = diffType(v, (C[3] >> (2 * 8)) & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 6;
    mask2 |= ((d1 & 2) >> 1) << 6;

    mask1 |= (d2 & 1) << 14;
    mask2 |= ((d2 & 2) >> 1) << 14;



    d1 = diffType(v, (C[0] >> (1 * 8)) & 0xff, th);
    d2 = diffType(v, (C[2] >> (1 * 8)) & 0xff, th);

    /*if ((d1 | d2) == 0)
        return;*/

    mask1 |= (d1 & 1) << 1;
    mask2 |= ((d1 & 2) >> 1) << 1;

    mask1 |= (d2 & 1) << 9;
    mask2 |= ((d2 & 2) >> 1) << 9;



    d1 = diffType(v, (C[0] >> (3 * 8)) & 0xff, th);
    d2 = diffType(v, (C[2] >> (3 * 8)) & 0xff, th);

    /*if ((d1 | d2) == 0)
        return;*/

    mask1 |= (d1 & 1) << 3;
    mask2 |= ((d1 & 2) >> 1) << 3;

    mask1 |= (d2 & 1) << 11;
    mask2 |= ((d2 & 2) >> 1) << 11;



    d1 = diffType(v, (C[1] >> (1 * 8)) & 0xff, th);
    d2 = diffType(v, (C[3] >> (1 * 8)) & 0xff, th);

    /*if ((d1 | d2) == 0)
        return;*/

    mask1 |= (d1 & 1) << 5;
    mask2 |= ((d1 & 2) >> 1) << 5;

    mask1 |= (d2 & 1) << 13;
    mask2 |= ((d2 & 2) >> 1) << 13;



    d1 = diffType(v, (C[1] >> (3 * 8)) & 0xff, th);
    d2 = diffType(v, (C[3] >> (3 * 8)) & 0xff, th);

    mask1 |= (d1 & 1) << 7;
    mask2 |= ((d1 & 2) >> 1) << 7;

    mask1 |= (d2 & 1) << 15;
    mask2 |= ((d2 & 2) >> 1) << 15;

    *mask1_ = mask1;
    *mask2_ = mask2;
}

inline uint popcnt(const uint i) {
    uint n;
    asm("popc.b32 %0, %1;" : "=r"(n) : "r" (i));
    return n;
}

  // 1 -> v > x + th
  // 2 -> v < x - th
  // 0 -> not a keypoint
inline bool isKeyPoint(int mask1, int mask2)
{
    return (popcnt(mask1) > 8 && (c_table[(mask1 >> 3) - 63] & (1 << (mask1 & 7)))) ||
           (popcnt(mask2) > 8 && (c_table[(mask2 >> 3) - 63] & (1 << (mask2 & 7))));
}

int cornerScore(const uint* C, const int v, const int threshold)
{
    // binary search in [threshold + 1, 255]

    int min = threshold + 1;
    int max = 255;

    while (min <= max)
    {
        const int mid = (min + max) >> 1;

        int mask1 = 0;
        int mask2 = 0;

        calcMask(C, v, mid, &mask1, &mask2);

        int isKp = (int)isKeyPoint(mask1, mask2);

        min = isKp * (mid + 1) + (isKp ^ 1) * min;
        max = (isKp ^ 1) * (mid - 1) + isKp * max;
    }

    return min - 1;
}

bool isKeyPoint2(
    const image2d_t img,
    const int i,
    const int j,
    const int threshold,
    int* scoreMat,
    int sRows,
    int sCols)
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

    int d1 = diffType(v, C[1] & 0xff, threshold);
    int d2 = diffType(v, C[3] & 0xff, threshold);

    if ((d1 | d2) == 0) {
        return false;
    }
    C[3] |= (uint)(PIXEL(i + 1, j - 3).x) << 8;
    C[0] |= (uint)(PIXEL(i + 1, j + 3).x) << (3 * 8);

    C[3] |= (uint)(PIXEL(i + 2, j - 2).x) << (2 * 8);
    C[0] |= (uint)(PIXEL(i + 2, j + 2).x) << (2 * 8);

    C[3] |= (uint)(PIXEL(i + 3, j - 1).x) << (3 * 8);
    C[0] |= (uint)(PIXEL(i + 3, j).x);
    C[0] |= (uint)(PIXEL(i + 3, j + 1).x) << 8;

    int mask1 = 0;
    int mask2 = 0;

    calcMask(C, v, threshold, &mask1, &mask2);
    if (i + j * sCols >= sRows * sCols) {
        return false;
    }
    if (isKeyPoint(mask1, mask2)) {
        scoreMat[i + j * sCols] = cornerScore(C, v, threshold);
        return true;
    }
    scoreMat[i + j * sCols] = 0;
    return false;
}

bool isMax(short2 loc, int* scoreMat, uint scoreMatCols) {
    int score = scoreMat[loc.x + loc.y * scoreMatCols];
    bool ismax =
        score > scoreMat[loc.x - 1 + (loc.y - 1) * scoreMatCols] &&
        score > scoreMat[loc.x     + (loc.y - 1) * scoreMatCols] &&
        score > scoreMat[loc.x + 1 + (loc.y - 1) * scoreMatCols] &&
        score > scoreMat[loc.x - 1 + (loc.y    ) * scoreMatCols] &&
        score > scoreMat[loc.x + 1 + (loc.y    ) * scoreMatCols] &&
        score > scoreMat[loc.x - 1 + (loc.y + 1) * scoreMatCols] &&
        score > scoreMat[loc.x     + (loc.y + 1) * scoreMatCols] &&
        score > scoreMat[loc.x + 1 + (loc.y + 1) * scoreMatCols];
    return ismax;
}

int factorial(int n) {
    //base case
    if(n == 0) {
        return 1;
    } else {
        return n * factorial(n-1);
    }
}

int fibbonacci(int n) {
    if(n == 0){
        return 0;
    } else if(n == 1) {
        return 1;
    } else {
        return (fibbonacci(n-1) + fibbonacci(n-2));
    }
}

void calcFibbonacci(int n) {
    printf("Fibbonacci of %d: " , n);
    for(int i = 0; i < n; ++i) {
        printf("%d ",fibbonacci(i));            
    }
}

__attribute__((reqd_work_group_size(32,8,1)))
__kernel void tileCalcKeypoints_kernel(
    __read_only const image2d_t img,
    __global short2* kpLoc, int klStep, int klOffset, int klRows, int klCols,
    __global float* kpScore, int ksStep, int ksOffset, int ksRows, int ksCols,
    __private uint maxKeypoints,
    __private uint highThreshold,
    __private uint lowThreshold,
    __global int* scoreMat, int sStep, int sOffset, int sRows, int sCols,
    __global int* debugMat, int dStep, int dOffset, int dRows, int dCols,
    __global uint* counterPtr, int cStep, int cOffset, int cRows, int cCols
    )
{
    // get_global_size(0) == 384,
    // get_global_size(1) == 24,
    // get_global_size(2) == 1,
    // get_local_size(0) == 32,
    // get_local_size(1) == 8,
    // get_local_size(2) == 1,
    // get_num_groups(1) == 3,
    // get_num_groups(2) == 1,
    // get_num_groups(3) == 1,
    // get_global_offset(0) == 0,
    // get_global_offset(1) == 0,
    // get_global_offset(2) == 0,
    // get_work_dim() == 2,

    // const struct  { uint x;            uint y; }
    //     gridDim = { get_global_size(0) / get_local_size(0),
    //                 get_global_size(1) / get_local_size(1) };
    // const struct   { uint x;            uint y; }
    //     blockDim = { get_local_size(0), get_local_size(1) };
    // const struct   { uint x;           uint y; }
    //     threadIdx = { get_global_id(0), get_global_id(1) };
    // debugMat[threadIdx.x + threadIdx.y * gridDim.x * blockDim.x] =
    //     threadIdx.x + threadIdx.y * gridDim.x * blockDim.x;

    const struct  { uint x;            uint y; }
        gridDim = { get_num_groups(0), get_num_groups(1) };
    const struct   { uint x;            uint y; }
        blockDim = { get_local_size(0), get_local_size(1) };
    const struct   { uint x;          uint y; }
        blockIdx = { get_group_id(0), get_group_id(1) };
    const struct    { uint x;          uint y; }
        threadIdx = { get_local_id(0), get_local_id(1) };
    debugMat[threadIdx.x + blockIdx.x * blockDim.x + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x]
        = threadIdx.x + blockIdx.x * blockDim.x + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

    const int j = threadIdx.x + blockIdx.x * blockDim.x + 3;
    const int i = (threadIdx.y + blockIdx.y * blockDim.y) * 4 + 3;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
    __local bool hasKp;
    if (tid == 0) {
        hasKp = false;
    }
    int imgRows = get_image_height(img);
    int imgCols = get_image_width(img);
    __private bool isKp[4] = {0, 0, 0, 0};
    for (int t = 0; t < 4; ++t) {
        if (i+t < imgRows - 3 && j < imgCols - 3) {
            isKp[t] = isKeyPoint2(img, i+t, j, highThreshold, scoreMat, sRows, sCols);
        }
    }

    // barrieer
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int t = 0; t < 4; ++t) {
        if (isKp[t]) {
            isKp[t] = false;
            short2 loc = (short2)(j, i+t);
            if (isMax(loc, scoreMat, sCols)) {
                int score = scoreMat[loc.x + loc.y * sCols];
                hasKp = true;
                const unsigned int ind = atomic_inc(counterPtr);
                // printf("$\n");
                if (ind < maxKeypoints) {
                    kpLoc[ind] = (short2)(j, i+t);
                    kpScore[ind] = (float)score;
                }
            }
        }
    }
    // barrieer
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    // if (hasKp) return;
    // if (threadIdx.x + blockIdx.x * blockDim.x + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x == 0) {
    //     calcFibbonacci(30);
    // }

    // lower the threshold and try again
    for (int t = 0; t < 4; ++t) {
        if (i+t < imgRows - 3 && j < imgCols - 3) {
            isKp[t] = isKeyPoint2(img, i+t, j, lowThreshold, scoreMat, sRows, sCols);
        }
    }
    // barrieer
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int t = 0; t < 4; ++t) {
        if (isKp[t]) {
            isKp[t] = false;
            short2 loc = (short2)(j, i+t);
            if (isMax(loc, scoreMat, sCols)) {
                int score = scoreMat[loc.x + loc.y * sCols];
                hasKp = true;
                const unsigned int ind = atomic_inc(counterPtr);
                // printf("€\n");
                if (ind < maxKeypoints) {
                    kpLoc[ind] = (short2)(j, i+t);
                    kpScore[ind] = (float)score;
                }
            }
        }
    }
}
