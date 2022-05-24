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

__constant int c_u_max[32] = {};

short2 make_short2(short x, short y)
{
    short2 res;
    res.x = x;
    res.y = y;
    return res;
}

typedef unsigned long long usize_t;

__kernel void IC_Angle_kernel(__global unsigned char *image, usize_t step, __global key_point_t *keypoints, const int npoints, const int half_k)
{
    __local int smem0[256];
    __local int smem1[256];

    const size_t thread_x = get_local_id(0);
    const size_t thread_y = get_local_id(1);
    const size_t block_x = get_group_id(0);
    const size_t blockDim_x = get_local_size(0);
    const size_t blockDim_y  = get_local_size(1);

    __local int* srow0 = smem0 + thread_y * blockDim_x;
    __local int* srow1 = smem1 + thread_y * blockDim_x;

    const int ptidx = block_x * blockDim_y + thread_y;

    if (ptidx < npoints)
    {
        int m_01 = 0, m_10 = 0;

        const short2 loc = make_short2(keypoints[ptidx].pt.x, keypoints[ptidx].pt.y);

        // Treat the center line differently, v=0
        for (int u = thread_x - half_k; u <= half_k; u += blockDim_x)
            m_10 += u * image[loc.y*step + loc.x + u];

        if (thread_x < m_10) {
            srow0[thread_x] = work_group_reduce_add(srow0[thread_x]);
        } else {
            work_group_reduce_add(0);
        }

        for (int v = 1; v <= half_k; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int m_sum = 0;
            const int d = c_u_max[v];

            for (int u = thread_x - d; u <= d; u += blockDim_x)
            {
                int val_plus = image[step*(loc.y + v) + loc.x + u];
                int val_minus = image[step*(loc.y - v) + loc.x + u];

                v_sum += (val_plus - val_minus);
                m_sum += u * (val_plus + val_minus);
            }

            // reduce<32>(smem_tuple(srow0, srow1), thrust::tie(v_sum, m_sum), threadIdx.x, thrust::make_tuple(op, op));
            if (thread_x < v_sum) {
                srow0[thread_x] = work_group_reduce_add(srow0[thread_x]);
            } else {
                work_group_reduce_add(0);
            }

            if (thread_x < m_sum) {
                srow1[thread_x] = work_group_reduce_add(srow1[thread_x]);
            } else {
                work_group_reduce_add(0);
            }

            m_10 += m_sum;
            m_01 += v * v_sum;
        }

        if (thread_x == 0)
        {
            //               vv  what is this ?
            //float kp_dir = ::atan2f((float)m_01, (float)m_10);
            float kp_dir = atan2((float)m_01, (float)m_10);
            kp_dir += (kp_dir < 0) * (2.0f * 3.141592f);
            kp_dir *= 180.0f / 3.141592f;

            keypoints[ptidx].angle = kp_dir;
        }
    }
}

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
