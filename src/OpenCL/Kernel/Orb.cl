typedef struct {
    float x;
    float y;
} Point2f;

typedef struct {
    unsigned char x;
    unsigned char y;
    unsigned char z;
} Point3b;

typedef struct {
    unsigned char *data;
    size_t step;
} PtrStepB;

typedef struct {
    int x;
    int y;
} Point2i;

typedef struct {
    Point2f pt;    //!< coordinates of the keyPoints
    float size;    //!< diameter of the meaningful keypoint neighborhood
    float angle;   //!< computed orientation of the keypoint (-1 if not applicable);
    float response;//!< the response by which the most strong keyPoints have been selected. Can be used for the further sorting or subsampling
    int octave;    //!< octave (pyramid layer) from which the keypoint has been extracted
    int class_id;  //!< object class (if the keyPoints need to be clustered by an object they belong to)
} key_point_t;

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

__kernel void calcOrb_kernel(
    __read_only const image2d_t image,
    __global char* keypoints,   int kStep, int kOffset, int kRows, int kCols,
    __global int* descriptors, int dStep, int dOffset, int dRows, int dCols)
{
    const size_t id = get_global_id(0) / get_local_size(0);
    const size_t tid = get_global_id(0) % get_local_size(0);
    __global key_point_t *kpt = (__global key_point_t *)keypoints;
    ushort2 loc = make_ushort2(kpt[id].pt.x, kpt[id].pt.y);
    Point2i *pattern = (Point2i *)c_pattern + 16 * tid;

    const float factorPI = (float)(CV_PI/180.f);
    float angle = (float)kpt[id].angle * factorPI;
    float cosA = cos(angle);
    float sinA = sin(angle);
    // __local int t[4];
    int t0, t1, val, idx;
    t0 = GET_VALUE(0); t1 = GET_VALUE(1);
    val = t0 < t1;
    t0 = GET_VALUE(2); t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4); t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6); t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8); t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10); t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12); t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14); t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;

    // zeros only (why?):
    descriptors[tid + id * get_local_size(0)] = (uchar)(1000 * (t1 - t0));

    // debuging:
    descriptors[0  + id * get_local_size(0)] = (uchar)(GET_VALUE(0 ));
    descriptors[1  + id * get_local_size(0)] = (uchar)(GET_VALUE(1 ));
    descriptors[2  + id * get_local_size(0)] = (uchar)(GET_VALUE(2 ));
    descriptors[3  + id * get_local_size(0)] = (uchar)(GET_VALUE(3 ));
    descriptors[4  + id * get_local_size(0)] = (uchar)(GET_VALUE(4 ));
    descriptors[5  + id * get_local_size(0)] = (uchar)(GET_VALUE(5 ));
    descriptors[6  + id * get_local_size(0)] = (uchar)(GET_VALUE(6 ));
    descriptors[7  + id * get_local_size(0)] = (uchar)(GET_VALUE(7 ));
    descriptors[8  + id * get_local_size(0)] = (uchar)(GET_VALUE(8 ));
    descriptors[9  + id * get_local_size(0)] = (uchar)(GET_VALUE(9 ));
    descriptors[10 + id * get_local_size(0)] = (uchar)(GET_VALUE(10));
    descriptors[11 + id * get_local_size(0)] = (uchar)(GET_VALUE(11));
    descriptors[12 + id * get_local_size(0)] = (uchar)(GET_VALUE(12));
    descriptors[13 + id * get_local_size(0)] = (uchar)(GET_VALUE(13));
    descriptors[14 + id * get_local_size(0)] = (uchar)(GET_VALUE(14));
    descriptors[15 + id * get_local_size(0)] = (uchar)(GET_VALUE(15));


    int4 test4;
    test4.xyzw = (int4)(4, 5, 6, 7);
    // for (idx = 0; idx < 16; idx += 2)
    // {
    //     int xIdx = loc.x + (int)(pattern[idx].x * cosA - pattern[idx].y * sinA);
    //     int yIdx = loc.y + (int)(pattern[idx].x * sinA + pattern[idx].y * cosA);
    //     float2 coord0 = (float2)((float)xIdx+0.5f, (float)yIdx+0.5f);
    //     float2 coord1 = (float2)((float)xIdx+1.5f, (float)yIdx+0.5f);
    //     t[0] = read_imagei(image, iSampler, coord0).x;
    //     t[1] = read_imagei(image, iSampler, coord1).x;
    //     val |= is_rhs_greater(t[0], t[1]) << (idx / 2);
    // }
}
