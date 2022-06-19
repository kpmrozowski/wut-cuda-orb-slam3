#include <boost/compute.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/platform.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <iostream>
#include <OpenCL/Manager.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <boost/compute/cl.hpp>
#include <boost/compute/types/struct.hpp>

typedef struct {
    float x;
    float y;
} point2f_t;

// adapt the chemistry::Atom class
BOOST_COMPUTE_ADAPT_STRUCT(point2f_t, point2f_t, (x, y))

typedef struct {
    point2f_t pt;    //!< coordinates of the keypoints
    float size;    //!< diameter of the meaningful keypoint neighborhood
    float angle;   //!< computed orientation of the keypoint (-1 if not applicable);
    float response;//!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
    int octave;    //!< octave (pyramid layer) from which the keypoint has been extracted
    int class_id;//!< object class (if the keypoints need to be clustered by an object they belong to)
} key_point_t;

// adapt the chemistry::Atom class
BOOST_COMPUTE_ADAPT_STRUCT(key_point_t, key_point_t, (pt, size, angle, response, octave, class_id))

key_point_t makeKp(float angle, float x, float y) {
    key_point_t kp;
    kp.size  = 1.0f;
    kp.angle = angle;
    kp.response = 0.1f;
    kp.class_id = 0;
    kp.octave = 0;
    kp.pt = {x, y};
    return kp;
}

std::array<key_point_t, 5> keyPoints{
        makeKp(0.1f, 1.0f, 1.0f),
        makeKp(0.3f, 3.0f, 4.0f),
        makeKp(0.2f, 2.0f, 1.0f),
        makeKp(0.4f, 0.0f, 1.0f),
        makeKp(0.4f, 1.0f, 2.0f),
};

void runKeyPointsKernel() {
    using ORB_SLAM3::opencl::Program;

    auto filename = "./datasets/MH01/cam0/data/1403636579763555584.png";    
    if (not std::filesystem::is_regular_file(filename)) {
        std::cout << "I am in: " << std::filesystem::current_path() << "\n";
        std::cout << "No such file or directory: " << filename << "\n";
        exit(1);
    }

    auto mat1 = cv::imread(filename);
    auto umat1 = mat1.getUMat(cv::ACCESS_RW);
    auto cl_buffer = static_cast<cl_mem>(umat1.handle(cv::ACCESS_RW));

    boost::compute::vector<key_point_t> gpuKeyPoints(5);
    boost::compute::copy(keyPoints.begin(), keyPoints.end(), gpuKeyPoints.begin());

    auto &manager = ORB_SLAM3::opencl::Manager::the();
    // __kernel void addBorder_kernel(__global key_point_t *keypoints, int npoints, int minBorderX, int minBorderY, int octave, int size) {
    auto start = manager.cv_run(Program::AngleKernel, "addBorder_kernel",  5, true, gpuKeyPoints, static_cast<int>(gpuKeyPoints.size()), 20, 20, 0, 5);
    // start.wait();
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

    auto start = manager.cv_run(Program::TestProgram, "squareVector",  256, true, gpuValues);
    // start.wait();

    compute::copy(gpuValues.begin(), gpuValues.end(), values.begin());

    std::cout << "after: \n";
    for (auto value : values) {
        std::cout << value << '\n';
    }
}

void runSimpleOCLProgram()
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

    auto start = manager.cv_run(Program::TestProgram, "squareVector",  256, true, gpuValues);
    // start.wait();

    compute::copy(gpuValues.begin(), gpuValues.end(), values.begin());

    std::cout << "after: \n";
    for (auto value : values) {
        std::cout << value << '\n';
    }
}

void boost_list_devices()
{
    boost::compute::device device =  boost::compute::system::default_device();
    std::cout << "hello from boost::compute" << std::endl;
    std::cout << "\tname: " << device.name() << std::endl;
    std::cout << "\tplatform: " << device.platform().name() << std::endl;
    std::cout << "\tdriver_version: " << device.driver_version() << std::endl;
    std::cout << "\tvendor: " << device.vendor() << std::endl;
    std::cout << "\tversion: " << device.version() << std::endl;
    std::cout << "\tclock_frequency: " << device.clock_frequency() << std::endl;
    std::cout << "\tcompute_units: " << device.compute_units() << std::endl;
    std::cout << "\tglobal_memory_size: " << device.global_memory_size() << std::endl;
    std::cout << "\tmax_work_group_size: " << device.max_work_group_size() << std::endl;
    std::cout << "\tid: " << device.id() << std::endl;
    std::cout << "\ttype: " << (device.type() == boost::compute::device::gpu ? "gpu" : "cpu") << std::endl;
}

void cv_cuda_list_devices()
{
    // Choose a random GPU
    auto device = cv::cuda::DeviceInfo();
    std::cout << "hello from cv::cuda:" << std::endl;
    std::cout << "\tname: " << device.name() << std::endl;
    std::cout << "\tversion: " << device.majorVersion() << "," << device.minorVersion() << std::endl;
    std::cout << "\tclockRate: " << device.clockRate() << std::endl;
    std::cout << "\tmemoryClockRate: " << device.memoryClockRate() << std::endl;
    std::cout << "\tasyncEngineCount: " << device.asyncEngineCount() << std::endl;
    std::cout << "\tmultiProcessorCount: " << device.multiProcessorCount() << std::endl;
    std::cout << "\ttotalMemory: " << device.totalMemory() << std::endl;
    std::cout << "\tcanMapHostMemory: " << device.canMapHostMemory() << std::endl;
    std::cout << "\twarpSize: " << device.warpSize() << std::endl;
    std::cout << "\tmaxGridSize: " << device.maxGridSize() << std::endl;
    std::cout << "\tl2CacheSize: " << device.l2CacheSize() << std::endl;
    std::cout << "\tid: " << device.deviceID() << std::endl;
    std::cout << "\ttype: " << (device.computeMode() == cv::cuda::DeviceInfo::ComputeModeDefault ? "ComputeModeDefault" : (
                                device.computeMode() == cv::cuda::DeviceInfo::ComputeModeExclusive ? "ComputeModeExclusive" : (
                                device.computeMode() == cv::cuda::DeviceInfo::ComputeModeExclusiveProcess ? "ComputeModeExclusiveProcess" : (
                                device.computeMode() == cv::cuda::DeviceInfo::ComputeModeProhibited ? "ComputeModeProhibited" : "cpu")))) << std::endl;
    int numDevices = cv::cuda::getCudaEnabledDeviceCount();
}

bool cv_ocl_list_devices()
{
    if (!cv::ocl::haveOpenCL())
    {
        std::cout << "OpenCL is not avaiable..." << std::endl;
        return 1;
    }
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
        std::cout << "Failed creating the context..." << std::endl;
        return 1;
    }

    // In OpenCV 3.0.0 beta, only a single device is detected.
    std::cout << context.ndevices() << " GPU devices are detected." << std::endl;
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        std::cout << "name                 : " << device.name() << std::endl;
        std::cout << "available            : " << device.available() << std::endl;
        std::cout << "imageSupport         : " << device.imageSupport() << std::endl;
        std::cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << std::endl;
        std::cout << std::endl;
    }
}

int main() {
    // boost_list_devices();
    // cv_cuda_list_devices();
    // cv_ocl_list_devices();
    // runSimpleOpenGLProgram();
    runKeyPointsKernel();
    std::cout << "success" << std::endl;
}
