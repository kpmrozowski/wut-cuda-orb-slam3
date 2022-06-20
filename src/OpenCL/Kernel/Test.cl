typedef struct {
    float x;
    float y;
} point2f_t;

typedef struct {
    point2f_t pt;    //!< coordinates of the keypoints
    float size;    //!< diameter of the meaningful keypoint neighborhood
    float angle;   //!< computed orientation of the keypoint (-1 if not applicable);
    float response;//!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
    int octave;    //!< octave (pyramid layer) from which the keypoint has been extracted
    int class_id;//!< object class (if the keypoints need to be clustered by an object they belong to)
} key_point_t;

__kernel void squareVector(__global unsigned char *image, __global key_point_t *keypoints, int npoints, int minBorderX, int minBorderY, int octave, int size) {
    int tid = get_global_id(0);
        if (tid >= npoints) {
            return;
        }
        keypoints[tid].class_id   = npoints;
        keypoints[tid].pt.x += minBorderX;
        keypoints[tid].pt.y += minBorderY;
        keypoints[tid].octave = octave;
        keypoints[tid].size   = size;
}

__kernel void squareVector2(
    const image2d_t image,
    int npoints,
    int minBorderX,
    int minBorderY,
    int octave,
    int size,
    __global char* keypoints, int dst_step, int dst_offset, int dst_rows, int dst_cols) {
    int tid = get_global_id(0);
        if (tid >= npoints) {
            return;
        }
        __global key_point_t *dstf = (__global key_point_t *)keypoints;
        dstf[tid].class_id    = npoints;
        dstf[tid].pt.x        += minBorderX;
        dstf[tid].pt.y        += minBorderY;
        dstf[tid].octave      = octave;
        dstf[tid].size        = size;
}
