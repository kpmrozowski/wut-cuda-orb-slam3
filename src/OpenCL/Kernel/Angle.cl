#include "OpenCL/Kernel/key_point.hpp"

__global static int c_u_max[16] = {15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3};

__kernel void IC_Angle_kernel(__global unsigned char *image, int iStep, int iOffset, int iRows, int iCols,
                              __global key_point_t *keypoints, int kStep, int kOffset, int kRows, int kCols,
                              const uint npoints, const uint half_k, __global int *debugMat, int dStep,
                              int dOffset, int dRows, int dCols)
{
    const size_t thread_x   = get_local_id(0);
    const size_t thread_y   = get_local_id(1);
    const size_t block_x    = get_group_id(0);
    const size_t blockDim_x = get_local_size(0);
    const size_t blockDim_y = get_local_size(1);

    const int ptidx = block_x * blockDim_y + thread_y;

    if (ptidx >= npoints)
        return;

    if (ptidx < npoints) {
        int m_01 = 0, m_10 = 0;
        const short2 loc = (short2) (keypoints[ptidx].pt.x, keypoints[ptidx].pt.y);

        // Treat the center line differently, v=0
        for (int u = thread_x - half_k; u <= half_k; u += blockDim_x)
            m_10 += u * image[loc.y * iCols + loc.x + u];

        for (int v = 1; v <= half_k; ++v) {
            // Proceed over the two lines
            int v_sum   = 0;
            int m_sum   = 0;
            const int d = c_u_max[v];

            for (int u = thread_x - d; u <= d; u += blockDim_x) {
                if ((iCols * (loc.y + v) + loc.x + u) >= iCols * iRows)  {
                    continue;
                }
                if ((iCols * (loc.y - v) + loc.x + u) < 0)   {
                    continue;
                }

                int val_plus  = image[iCols * (loc.y + v) + loc.x + u];
                int val_minus = image[iCols * (loc.y - v) + loc.x + u];

                v_sum += (val_plus - val_minus);
                m_sum += u * (val_plus + val_minus);
            }
            m_10 += m_sum;
            m_01 += v * v_sum;
        }

        if (thread_x == 0) {
            float kp_dir = atan2((float) m_01, (float) m_10);
            kp_dir += (kp_dir < 0) * (2.0f * 3.141592f);
            kp_dir *= 180.0f / 3.141592f;
            keypoints[ptidx].angle = kp_dir;
        }
    }
}
