#include <boost/compute.hpp>
#include <OpenCL/Orb.hpp>
#include <Utils.hpp>

namespace ORB_SLAM3::opencl {


__constant__ unsigned char c_pattern[sizeof(Point) * 512];

void GpuOrb::loadPattern(const Point * _pattern) {
checkCudaErrors( cudaMemcpyToSymbol(c_pattern, _pattern, sizeof(Point) * 512) );
}

#define GET_VALUE(idx) \
    image(loc.y + __float2int_rn(pattern[idx].x * b + pattern[idx].y * a), \
          loc.x + __float2int_rn(pattern[idx].x * a - pattern[idx].y * b))

// clang-format off
constexpr auto g_openCLProgram = BOOST_COMPUTE_STRINGIZE_SOURCE(
        typedef struct {
            float x;
            float y;
        } Point2f;

        typedef struct {
            Point2f pt;    //!< coordinates of the keyPoints
            float size;    //!< diameter of the meaningful keypoint neighborhood
            float angle;   //!< computed orientation of the keypoint (-1 if not applicable);
            float response;//!< the response by which the most strong keyPoints have been selected. Can be used for the further sorting or subsampling
            int octave;    //!< octave (pyramid layer) from which the keypoint has been extracted
            int class_id; //!< object class (if the keyPoints need to be clustered by an object they belong to)
        } KeyPoint;

__kernel void calcOrb_kernel(const Image image, KeyPoint * keyPoints, const int nPoints, PtrStepb descriptors) {
int id = blockIdx.x;
        int tid = get_global_id(0);
        const int groupId = get_group_id(0); // upto 8
        if (id >= nPoints) return;

        const KeyPoint &kpt = keypoints[id];
        short2 loc = make_short2(kpt.pt.x, kpt.pt.y);
        const Point * pattern = ((Point *)c_pattern) + 16 * tid;

        uchar * desc = descriptors.ptr(id);
        const float factorPI = (float)(CV_PI/180.f);
        float angle = (float)kpt.angle * factorPI;
        float a = (float)cosf(angle), b = (float)sinf(angle);

        int t0, t1, val;
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

        desc[tid] = (uchar)val;
    }
});
// clang-format on


namespace compute = boost::compute;

void GpuOrb::launch_async(cv::InputArray _image, const cv::KeyPoint *_keyPoints, const int nPoints)
{
    if (nPoints == 0) {
        POP_RANGE;
        return;
    }
    auto gpu             = compute::system::default_device();
    cv::UMat imageMatrix = _image.getUMat();
    assert(imageMatrix.type() == CV_8UC3);
    auto arg = cv::ocl::KernelArg::ReadWrite(imageMatrix, _image.channels());

    compute::context ctx{gpu};
    compute::copy(_keyPoints, _keyPoints + nPoints, m_gpuKeyPoints.begin());
    m_gpuDesc = m_gpuDescriptors.rowRange(0, nPoints);
    m_gpuDesc.setTo(cv::Scalar::all(0));
    cv::ocl::Image2D image(imageMatrix);
    auto ptr = image.ptr();
}
}