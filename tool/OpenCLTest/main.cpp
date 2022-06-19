#include <boost/compute.hpp>
#include <fmt/core.h>
#include <iostream>
#include <OpenCL/Manager.hpp>
#include <opencv2/imgcodecs.hpp>

cv::KeyPoint makeKp(float angle, float x, float y) {
    cv::KeyPoint kp;
    kp.size  = 1.0f;
    kp.angle = angle;
    kp.response = 0.1f;
    kp.class_id = 0;
    kp.octave = 0;
    kp.pt = {x, y};
    return kp;
}

std::array<cv::KeyPoint, 5> keyPoints{
        makeKp(0.1f, 1.0f, 1.0f),
        makeKp(0.3f, 3.0f, 4.0f),
        makeKp(0.2f, 2.0f, 1.0f),
        makeKp(0.4f, 0.0f, 1.0f),
        makeKp(0.4f, 1.0f, 2.0f),
};

void runKeyPointsKernel() {
    using ORB_SLAM3::opencl::Program;

    auto mat1 = cv::imread("./datasets/MH01/mav0/cam0/data/1403636579763555584.png");
    auto umat1 = mat1.getUMat(cv::ACCESS_RW);
    auto cl_buffer = static_cast<cl_mem>(umat1.handle(cv::ACCESS_RW));

    boost::compute::vector<cv::KeyPoint> gpuKeyPoints(5);
    boost::compute::copy(keyPoints.begin(), keyPoints.end(), gpuKeyPoints.begin());

    auto &manager = ORB_SLAM3::opencl::Manager::the();
    // __kernel void addBorder_kernel(__global key_point_t *keypoints, int npoints, int minBorderX, int minBorderY, int octave, int size) {
    auto start = manager.run(Program::AngleKernel, "addBorder_kernel",  5, gpuKeyPoints, static_cast<int>(gpuKeyPoints.size()), 20, 20, 0, 5);
    start.wait();
};

void runSimpleOpenGLProgram()
{
    using ORB_SLAM3::opencl::Program;
    namespace compute = boost::compute;

    auto &manager = ORB_SLAM3::opencl::Manager::the();

    std::vector<int> values(256);
    std::iota(values.begin(), values.end(), 0);

    std::cout << "before: \n";
    for (auto value : values) {
        std::cout << value << '\n';
    }

    compute::vector<int> gpuValues(256);
    compute::copy(values.begin(), values.end(), gpuValues.begin());

    auto start = manager.run(Program::TestProgram, "squareVector",  256, gpuValues);
    start.wait();

    compute::copy(gpuValues.begin(), gpuValues.end(), values.begin());

    std::cout << "after: \n";
    for (auto value : values) {
        std::cout << value << '\n';
    }
}

int main() {
    runSimpleOpenGLProgram();
//    runKeyPointsKernel();
}