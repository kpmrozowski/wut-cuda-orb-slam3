#include "OpenCL/Manager.hpp"
#include <opencv2/core/mat.hpp>
#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/compute.hpp>
#include <boost/compute/types/struct.hpp>
#include <boost/range/combine.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <span>

using ORB_SLAM3::opencl::Program;

typedef struct
{
    float x;
    float y;
} point2f_t;

BOOST_COMPUTE_ADAPT_STRUCT(point2f_t, point2f_t, (x, y))

typedef struct
{
    point2f_t pt;  //!< coordinates of the keypoints
    float size;    //!< diameter of the meaningful keypoint neighborhood
    float angle;   //!< computed orientation of the keypoint (-1 if not applicable);
    float response;//!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
    int octave;    //!< octave (pyramid layer) from which the keypoint has been extracted
    int class_id;  //!< object class (if the keypoints need to be clustered by an object they belong to)
} key_point_t;

BOOST_COMPUTE_ADAPT_STRUCT(key_point_t, key_point_t, (pt, size, angle, response, octave, class_id))

key_point_t makeKp(float angle, float x, float y)
{
    key_point_t kp;
    kp.size     = 1.0f;
    kp.angle    = angle;
    kp.response = 0.1f;
    kp.class_id = 0;
    kp.octave   = 0;
    kp.pt       = {x, y};
    return kp;
}

TEST(list_boost_devices, OpenCLTest)
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

TEST(list_cv_ocl_devices, OpenCLTest)
{
    ASSERT_TRUE(cv::ocl::haveOpenCL());
    cv::ocl::Context context;
    ASSERT_TRUE(context.create(cv::ocl::Device::TYPE_GPU));

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

template<typename T, int CV_TYPE = CV_8SC1>
class CvVector
{
    std::vector<T> m_vec;
    cv::Mat m_mat_before;
    size_t m_size;
    cv::UMat m_umat;
    std::optional<cv::Mat> m_mat = std::nullopt;
    bool synchronized = true;

public:
    explicit CvVector(std::vector<T>&& vec)
     : m_vec(vec)
     , m_size(m_vec.size())
     , m_mat_before(1, m_vec.size() * sizeof(T), CV_TYPE, m_vec.data())
    {
    }

    [[nodiscard]] constexpr size_t size()
    {
        return m_size;
    }

    [[nodiscard]] std::span<T> before()
    {
        return std::span<T>{reinterpret_cast<T*>(m_mat_before.data), m_size};
    }

    [[nodiscard]] cv::UMat& umat()
    {
        synchronized = false;
        m_umat = m_mat_before.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        return m_umat;
    }

    [[nodiscard]] cv::ocl::KernelArg kernelArg()
    {
        synchronized = false;
        m_umat = m_mat_before.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        return cv::ocl::KernelArg::ReadWrite(m_umat);
    }

    std::span<T> result()
    {
        if (not m_mat.has_value()) {
            m_mat = m_umat.getMat(cv::ACCESS_READ);
            return std::span<T>{reinterpret_cast<T*>(m_mat_before.data), m_size};
        }
        return {
            reinterpret_cast<T*>(m_mat.value().data),
            m_size
        };
    }
};

/* A SMALL ESENSE OF ABOVE CLASS
template<typename Tvec>
cv::UMat vec2umat(Tvec& vec)
{
    return cv::Mat{
        1, static_cast<int>(vec.size() * sizeof(key_point_t)), CV_8SC1, vec.data()
    }.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
}

std::span<key_point_t> umat2vec(cv::UMat& umat, size_t size)
{
    return {
        reinterpret_cast<key_point_t*>(umat.getMat(cv::ACCESS_READ).data),
        size
    };
}
*/

cv::Mat& load_sample_image()
{
    // auto filename = "./datasets/MH01/mav0/cam0/data/1403636579763555584.png";
    auto filename = "/shm/datasets/Kitti/sequences/03/image_0/000000.png";
    if (not std::filesystem::is_regular_file(filename)) {
        throw std::runtime_error("image does not exist");
    } else {
        std::cout << "image exists\n";
    }
    static cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    image.convertTo(image, CV_32F, 1.0 / 255);
    return image;
}

TEST(runKeyPointsKernel2, OpenCLTest)
{
    auto &manager = ORB_SLAM3::opencl::Manager::the();
    cv::Mat& image = load_sample_image();
    cv::UMat umat_src  = image.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat umat_dest = cv::UMat(image.size(), cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::ocl::Image2D image2d{umat_src};

    CvVector<key_point_t> cvKeyPoints{{
            makeKp(0.1f, 1.0f, 1.0f), makeKp(0.3f, 3.0f, 4.0f), makeKp(0.2f, 2.0f, 1.0f),
            makeKp(0.4f, 0.0f, 1.0f), makeKp(0.4f, 1.0f, 2.0f),}};

    std::vector<size_t> gridDim{5};
    auto start = manager.cv_run<1>(
        Program::TestProgram,
        "squareVector2",
        gridDim.data(),
        true,
        /*image2d_t*/ image2d,
        /*npoints*/ 5,
        /*minBorderX*/ 20,
        /*minBorderY*/ 20,
        /*octave*/ 0,
        /*size*/ 5,
        /*keypoints*/ cvKeyPoints.kernelArg());

    ASSERT_TRUE(start);

    for (const auto &[lhs, rhs] : boost::combine(cvKeyPoints.before(), cvKeyPoints.result()))
    {
        ASSERT_EQ(5, rhs.class_id);
        std::cout << 5 << " == " << rhs.class_id << "\n";
    }
}

TEST(runKeyPointsKernel, OpenCLTest)
{
    auto &manager = ORB_SLAM3::opencl::Manager::the();

    cv::Mat& image = load_sample_image();
    cv::UMat umat_src = image.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    auto umat_dest = cv::UMat(image.size(), cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::ocl::Image2D image2d{umat_src};
    //    auto cl_buffer = static_cast<cl_mem>(umat_src.handle(cv::ACCESS_RW));

    std::array<key_point_t, 5> keyPoints{
            makeKp(0.1f, 1.0f, 1.0f), makeKp(0.3f, 3.0f, 4.0f), makeKp(0.2f, 2.0f, 1.0f),
            makeKp(0.4f, 0.0f, 1.0f), makeKp(0.4f, 1.0f, 2.0f),
    };

    boost::compute::vector<key_point_t> gpuKeyPoints(5);
    boost::compute::copy(keyPoints.begin(), keyPoints.end(), gpuKeyPoints.begin());

    // __kernel void addBorder_kernel(__global key_point_t *keypoints, int npoints, int minBorderX, int minBorderY, int octave, int size) {
    std::vector<size_t> gridDim{5};
    auto start = manager.cv_run<1>(
        Program::TestProgram,
        "squareVector",
        gridDim.data(),
        true,
        image2d,
        gpuKeyPoints.get_buffer().get(),
        /*npoints*/ 5,
        /*minBorderX*/ 20,
        /*minBorderY*/ 20,
        /*octave*/ 0,
        /*size*/ 5
    );

    ASSERT_TRUE(start);

    std::vector<key_point_t> values_out(5);
    boost::compute::copy(gpuKeyPoints.begin(), gpuKeyPoints.end(), values_out.begin());

    for (const auto &[lhs, rhs] : boost::combine(keyPoints, values_out)) {
        ASSERT_EQ(5, rhs.class_id);
    }
}

TEST(runSimpleOpenGLProgram, OpenCLTest)
{
    namespace compute = boost::compute;

    //    std::string source = BOOST_COMPUTE_STRINGIZE_SOURCE(
    //        __kernel void squareVector(__global int *data) {
    //            const int globalId   = get_global_id(0);
    //            const int value = data[globalId];
    //            data[globalId] = value * value;
    //        }
    //    );

    auto &manager = ORB_SLAM3::opencl::Manager::the();

    std::vector<key_point_t> values(256), values_out(256);
    std::for_each(values.begin(), values.end(), [i = 0](key_point_t &v) mutable { v.class_id = ++i; });
    //    std::iota(values.begin(), values.end(), 0);

    compute::vector<key_point_t> gpuValues(256);
    compute::copy(values.begin(), values.end(), gpuValues.begin());

    auto start = manager.run(
        Program::TestProgram,
        "squareVector",
        256,
        gpuValues,
        /*npoints*/ 5,
        /*minBorderX*/ 20,
        /*minBorderY*/ 20,
        /*octave*/ 0,
        /*size*/ 5
    );
    start.wait();

    compute::copy(gpuValues.begin(), gpuValues.end(), values_out.begin());

    for (const auto &[lhs, rhs] : boost::combine(values, values_out)) {
        ASSERT_EQ(lhs.class_id * 5, rhs.class_id);
    }
}

TEST(runCalcOrbKernel, OpenCLTest)
{
    auto &manager = ORB_SLAM3::opencl::Manager::the();
    cv::Mat& image = load_sample_image();
    image.convertTo(image, CV_32F, 1.0 / 255);
    cv::UMat umat_src  = image.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::ocl::Image2D image2d{umat_src};

    CvVector<key_point_t> cvKeyPoints{{
            makeKp(0.1f, 1.0f, 1.0f), makeKp(0.3f, 3.0f, 4.0f), makeKp(0.2f, 2.0f, 1.0f),
            makeKp(0.4f, 0.0f, 1.0f), makeKp(0.4f, 1.0f, 2.0f),}};

    CvVector cvDescriptors{std::vector<uint>(cvKeyPoints.size() * 32)};

    std::vector<size_t> gridDim{32 * cvKeyPoints.size()};
    std::vector<size_t> blockDim{32};
    auto start = manager.cv_run<1>(
        Program::OrbKernel,
        gridDim.data(),
        blockDim.data(),
        true,
        /*image2d_t */ image2d,
        /*char* */ cvKeyPoints.umat(),
        /*image2d_t */ cvDescriptors.umat());

    ASSERT_TRUE(start);

    std::cout << "cvDescriptors = [\n";
    for (int32_t i = 0; i < cvKeyPoints.size(); ++i)
    {
        for (int32_t j = 0; j < 32; ++j)
            std::cout << cvDescriptors.result()[j + 32 * i] << ", ";
        std::cout << "\n";
    }
    std::cout << "]\n";
}

struct short2
{
    short x;
    short y;
};

/* 
################### BEFORE: ###################

dimGrid: 38, 11, 1
dimBlock: 32, 8, 1
image: 343, 1210, 
image.type: 0
kpLoc: 0, 0, 0, 0
kpScore: 0, 0
maxKeypoints: 10000
Threshold: 20, 7
scoreMat: 343, 1210
counterPtr: 0

################### AFTER: ###################

dimGrid: 38, 11, 1
dimBlock: 32, 8, 1
image: 343, 1210, 
image.type: 0
kpLoc: 191, 3, 191, 10
kpScore: 7, 9
maxKeypoints: 10000
Threshold: 20, 7
scoreMat: 343, 1210
counterPtr: 5927
 */

TEST(runTileCalcKeypointsKernel, OpenCLTest)
{
    auto &manager = ORB_SLAM3::opencl::Manager::the();
    cv::Mat& image = load_sample_image();
    // cv::imshow("", image);
    // cv::waitKey();
    image.convertTo(image, CV_32F, 1.0 / 255);
    cv::UMat umat_src  = image.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::ocl::Image2D image2d{umat_src};

    uint maxKeypoints = 10'000;
    struct { uint x = 12; uint y = 3; } dimGrid;
    struct { uint x = 32; uint y = 8; } dimBlock;
    CvVector kpLoc{std::vector<short2>(maxKeypoints)};
    CvVector kpScore{std::vector<float>(maxKeypoints)};
    uint highThreshold = 20, lowThreshold = 7;
    uint scoreMat_rows = 343, scoreMatCols = 1210;
    CvVector scoreMat{std::vector<int>(scoreMat_rows * scoreMatCols)};
    CvVector counterPtr{std::vector<uint>(1)};
    CvVector debugMat{std::vector<uint>(dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y)};

    std::cout << "\n################### BEFORE: ###################\n" <<
        "\ndimGrid: " << dimGrid.x << ", " << dimGrid.y <<
        "\ndimBlock: " << dimBlock.x << ", " << dimBlock.y <<
        "\nimage: " << image.rows << ", " << image.cols << ", " <<
        "\nimage.type: " << image.type() <<
        "\nkpLoc: ";
    for (int i = 0; i < 20; ++i) {
        std::cout << "(" << kpLoc.before()[i].x << ", " << kpLoc.before()[i].y << "), ";
    }
    std::cout <<
        "\nkpScore: ";
    for (int i = 0; i < 20; ++i) {
        std::cout << kpScore.before()[i] << ", ";
    }
    std::cout <<
        "\nmaxKeypoints: " << maxKeypoints <<
        "\nThreshold: " << highThreshold << ", " << lowThreshold <<
        "\nscoreMat: " << scoreMat_rows << ", " << scoreMatCols <<
        "\ncounter_ptr: " << counterPtr.before()[0] << std::endl;

    std::vector<size_t> gridDim{dimGrid.x * dimBlock.x, dimGrid.y * dimBlock.y};
    std::vector<size_t> blockDim{dimBlock.x, dimBlock.y};
    std::cout << "\nrunning kernel";
    auto start = manager.cv_run<2>(
        Program::TileCalcKeypointsKernel,
        gridDim.data(),
        blockDim.data(),
        true,
        /*image2d_t */ image2d,
        /*int */ image.rows,
        /*int */ image.cols,
        /*short2* */ kpLoc.kernelArg(),
        /*float* */ kpScore.kernelArg(),
        /*uint */ maxKeypoints,
        /*uint */ highThreshold,
        /*uint */ lowThreshold,
        /*int* */ scoreMat.kernelArg(),
        /*uint */ scoreMatCols,
        /*uint* */ counterPtr.kernelArg(),
        /*uint* */ debugMat.kernelArg()
    );
    std::cout << "\nkernel finished\n";
// image, kpLoc, kpScore, maxKeypoints, highThreshold, lowThreshold, scoreMat, counterPtr
    ASSERT_TRUE(start);

    std::cout <<  "################### AFTER: ###################\n" <<
        "\ndimGrid: " << dimGrid.x << ", " << dimGrid.y <<
        "\ndimBlock: " << dimBlock.x << ", " << dimBlock.y <<
        "\nimage: " << image.rows << ", " << image.cols << ", " <<
        "\nimage.type: " << image.type() <<
        "\nkpLoc: ";
    for (int i = 0; i < 20; ++i) {
        std::cout << kpLoc.result()[i].x << ", " << kpLoc.result()[i].y << ", ";
    }
    std::cout << "\nkpScore: ";
    for (int i = 0; i < 20; ++i) {
        std::cout << kpScore.result()[i] << ", ";
    }
    std::cout <<
        "\nmaxKeypoints: " << maxKeypoints <<
        "\nThreshold: " << highThreshold << ", " << lowThreshold <<
        "\nscoreMat: " << scoreMat_rows << ", " << scoreMatCols <<
        "\ncounter_ptr: " << counterPtr.result()[0] << std::endl;
    auto debugMatResult = debugMat.result();
    for (int i = 0; i < dimGrid.x * dimGrid.y; ++i) {
        for (int j = 0; j < dimBlock.x * dimBlock.y; ++j) {
            std::cout << debugMatResult[j + i * dimBlock.x * dimBlock.y] << ",";
        }
        std::cout << "\n";
    }
    std::cout << "\ndimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x * dimBlock.y * dimBlock.z = "
        << dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y << "\n";
}
