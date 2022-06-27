#include <boost/compute.hpp>
#include <OpenCL/Fast.hpp>
#include <OpenCL/Manager.hpp>

namespace ORB_SLAM3::opencl {

void IC_Angle::launch_async(const cv::_InputArray &_image, cv::KeyPoint *_keyPoints, int nPoints, int half_k,
                            int minBorderX, int minBorderY, int octave, int size)
{
    namespace compute = boost::compute;
    auto imageMatrix  = _image.getUMat();

    auto &gpuMan = Benchmark::the();

    compute::vector<cv::KeyPoint> gpuKeyPoints(nPoints);
    compute::copy(_keyPoints, _keyPoints + nPoints, gpuKeyPoints.begin());

    auto addBorderRes = gpuMan.run(Program::AngleKernel, "addBorder_kernel", nPoints, minBorderX, minBorderY, octave, size);

    auto angleRes = gpuMan.run(Program::AngleKernel, "IC_Angle_kernel", nPoints, minBorderX, minBorderY, octave, size);

    addBorderRes.wait();
    angleRes.wait();

    compute::copy(_keyPoints, _keyPoints + nPoints, gpuKeyPoints.begin());
}

}// namespace ORB_SLAM3::opencl
