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
} KeyPoint;

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

__global unsigned char c_pattern[sizeof(Point2i) * 512] = {};
#define CV_PI 3.141592f

__kernel void calcOrb_kernel(__global const Point3b *image,
                            __global KeyPoint *keyPoints,
                            const int nPoints,
                            __global unsigned char *descriptors,
                            __const int width)
{
    if (nPoints == 0) {
      return;
    }
    const size_t id = get_group_id(0);
    const size_t tid = get_local_id(0);
    const KeyPoint kpt = keyPoints[id];
    ushort2 loc = make_ushort2(kpt.pt.x, kpt.pt.y);
    Point2i *pattern = (Point2i *)c_pattern + 16 * tid;

    unsigned char * desc = descriptors + id;
    const float factorPI = (float)(CV_PI/180.f);
    float angle = (float)kpt.angle * factorPI;
    float cosA = cos(angle);
    float sinA = sin(angle);
    Point3b t0, t1;
    int val, idx;
    for (idx = 0; idx < 16; idx += 2)
    {
        int xIdx = loc[0] + (int)(pattern[idx].x * cosA - pattern[idx].y * sinA);
        int yIdx = loc[1] + (int)(pattern[idx].x * sinA + pattern[idx].y * cosA);
        t0 = image[xIdx + yIdx * width];
        t1 = image[xIdx + yIdx * width + 1];
        val |= is_rhs_greater(t0, t1) << (idx / 2);
    }
}
