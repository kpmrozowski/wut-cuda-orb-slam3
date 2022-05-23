#include <OpenCL/Manager.hpp>

namespace ORB_SLAM3::opencl {

extern const char *g_testProgram;
extern const char *g_angleProgram;

static auto makeContext() {
    if (!cv::ocl::haveOpenCL()) {
        std::exit(1);
    }

    cv::ocl::Context ctx;
    if (!ctx.create(cv::ocl::Device::TYPE_GPU))  {
        std::exit(1);
    }

    return ctx;
}

Manager::Manager() :
    m_context(makeContext()),
    m_device(m_context.device(0)),
    m_queue(m_context, m_device),
    m_workGroupSize(m_device.maxWorkGroupSize()),
    m_programs{cv::ocl::Program(cv::ocl::ProgramSource(g_testProgram), "", m_errorMsg),
               cv::ocl::Program(cv::ocl::ProgramSource(g_angleProgram), "", m_errorMsg)}
{
}

Manager &Manager::the()
{
    static Manager manager;
    return manager;
}

const char *g_testProgram = BOOST_COMPUTE_STRINGIZE_SOURCE(
        struct test {
            int foo;
            int bar;
        };

        typedef struct test test_t;

        __kernel void squareVector(__global test_t *values) {
            const int globalId   = get_global_id(0);// groupId * size  + local
            const test_t value   = values[globalId];
            values[globalId].foo = value.bar - value.foo;
        });

const char *g_angleProgram = BOOST_COMPUTE_STRINGIZE_SOURCE(
        typedef struct {
            float x;
            float y;
        } Point2f;

        typedef struct {
            Point2f pt;    //!< coordinates of the keypoints
            float size;    //!< diameter of the meaningful keypoint neighborhood
            float angle;   //!< computed orientation of the keypoint (-1 if not applicable);
            float response;//!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
            int octave;    //!< octave (pyramid layer) from which the keypoint has been extracted
            int class_id;//!< object class (if the keypoints need to be clustered by an object they belong to)
        } KeyPoint;

        __kernel void IC_Angle_kernel(const PtrStepb image, KeyPoint * keypoints, const int npoints, const int half_k)
        {
            __local int smem0[8 * 32];
            __local int smem1[8 * 32];

            int* srow0 = smem0 + threadIdx.y * blockDim.x;
            int* srow1 = smem1 + threadIdx.y * blockDim.x;

            cv::cuda::device::plus<int> op;

            const int ptidx = blockIdx.x * blockDim.y + threadIdx.y;

            if (ptidx < npoints)
            {
                int m_01 = 0, m_10 = 0;

                const short2 loc = make_short2(keypoints[ptidx].pt.x, keypoints[ptidx].pt.y);

                // Treat the center line differently, v=0
                for (int u = threadIdx.x - half_k; u <= half_k; u += blockDim.x)
                    m_10 += u * image(loc.y, loc.x + u);

                reduce<32>(srow0, m_10, threadIdx.x, op);

                for (int v = 1; v <= half_k; ++v)
                {
                    // Proceed over the two lines
                    int v_sum = 0;
                    int m_sum = 0;
                    const int d = c_u_max[v];

                    for (int u = threadIdx.x - d; u <= d; u += blockDim.x)
                    {
                        int val_plus = image(loc.y + v, loc.x + u);
                        int val_minus = image(loc.y - v, loc.x + u);

                        v_sum += (val_plus - val_minus);
                        m_sum += u * (val_plus + val_minus);
                    }

                    reduce<32>(smem_tuple(srow0, srow1), thrust::tie(v_sum, m_sum), threadIdx.x, thrust::make_tuple(op, op));

                    m_10 += m_sum;
                    m_01 += v * v_sum;
                }

                if (threadIdx.x == 0)
                {
                    //               vv  what is this ?
                    //float kp_dir = ::atan2f((float)m_01, (float)m_10);
                    float kp_dir = atan2f((float)m_01, (float)m_10);
                    kp_dir += (kp_dir < 0) * (2.0f * CV_PI_F);
                    kp_dir *= 180.0f / CV_PI_F;

                    keypoints[ptidx].angle = kp_dir;
                }
            }
        }

        __kernel void addBorder_kernel(__global KeyPoint *keypoints, int npoints, int minBorderX,
                                       int minBorderY, int octave, int size) {
            int tid = get_global_id(0);
            if (tid >= npoints) {
                return;
            }
            keypoints[tid].pt.x += minBorderX;
            keypoints[tid].pt.y += minBorderY;
            keypoints[tid].octave = octave;
            keypoints[tid].size   = size;
        });

}// namespace ORB_SLAM3::opencl
