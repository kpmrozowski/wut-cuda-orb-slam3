#include <boost/compute.hpp>
#include <OpenCL/Orb.hpp>
#include <Utils.hpp>
#include <OpenCL/Manager.hpp>

namespace ORB_SLAM3::opencl {

GpuOrb::GpuOrb(int maxKeyPoints) : m_maxKeyPoints(maxKeyPoints), m_gpuDescriptors(maxKeyPoints, 32, CV_8UC1) {}

constexpr auto g_openCLProgram2 = R"(
#include ""

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
    int class_id;  //!< object class (if the keyPoints need to be clustered by an object they belong to)
} KeyPoint;

)";

namespace compute = boost::compute;

void GpuOrb::launch_async(cv::InputArray _image, const cv::KeyPoint *_keyPoints, const int nPoints)
{
    if (nPoints == 0) {
        POP_RANGE;
        return;
    }
    auto &gpuMan = Benchmark::the();
    auto gpu = compute::system::default_device();
    assert(_image.type() == CV_8UC3);
    cv::UMat gpuImageSrc = _image.getUMat(cv::ACCESS_READ);

    compute::context ctx{gpu};
    compute::copy(_keyPoints, _keyPoints + nPoints, m_gpuKeyPoints.begin());
    m_gpuDesc = m_gpuDescriptors.rowRange(0, nPoints);
    m_gpuDesc.setTo(cv::Scalar::all(0));
    cv::ocl::Image2D image(gpuImageSrc);
    cv::ocl::Image2D desc(m_gpuDesc);
//    gpuMan.cv_run(Program::OrbKernel, "calcOrb_kernel",  32*8, 1, true,
//                    gpuImageSrc, m_gpuKeyPoints, nPoints, m_gpuDescriptors, gpuImageSrc.cols);
//    gpuMan.run
//    auto ptr = image.ptr();
}
}