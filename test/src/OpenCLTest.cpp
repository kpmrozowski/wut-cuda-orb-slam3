#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/compute.hpp>
#include <boost/compute/types/struct.hpp>
#include <boost/range/combine.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <iostream>
#include <OpenCL/Manager.hpp>
#include <opencv2/imgcodecs.hpp>
#include <span>

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

template<typename T>
class CvVector
{
    size_t m_size;
    std::vector<T> m_vec;
    cv::Mat m_mat_before;
    cv::UMat m_umat;
    std::optional<cv::Mat> m_mat = std::nullopt;
    bool synchronized = true;

public:
    CvVector(std::vector<T> vec)
     : m_vec(vec)
     , m_size(vec.size())
     , m_mat_before(1, vec.size() * sizeof(key_point_t), CV_8SC1, vec.data())
    {
    }

    [[nodiscard]] std::span<T> before()
    {
        return std::span<key_point_t>{reinterpret_cast<key_point_t*>(m_mat_before.data), m_size};
    }

    [[nodiscard]] cv::ocl::KernelArg kernelArg()
    {
        synchronized = false;
        m_umat = m_mat_before.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        return cv::ocl::KernelArg::ReadWrite(m_umat);
    }

    std::span<key_point_t> result()
    {
        if (not m_mat.has_value()) {
            m_mat = m_umat.getMat(cv::ACCESS_READ);
            return std::span<key_point_t>{reinterpret_cast<key_point_t*>(m_mat_before.data), m_size};
        }
        return {
            reinterpret_cast<key_point_t*>(m_mat.value().data),
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

TEST(runKeyPointsKernel2, OpenCLTest)
{
    using ORB_SLAM3::opencl::Program;
    auto &manager = ORB_SLAM3::opencl::Manager::the();
    auto filename = "./datasets/MH01/mav0/cam0/data/1403636579763555584.png";
    if (not std::filesystem::is_regular_file(filename)) {
        throw std::runtime_error("image does not exist");
    } else {
        std::cout << "image exists\n";
    }
    cv::Mat mat = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    mat.convertTo(mat, CV_32F, 1.0 / 255);
    cv::UMat umat_src  = mat.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat umat_dest = cv::UMat(mat.size(), cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::ocl::Image2D image{umat_src};

    std::vector<key_point_t> keyPoints{
            makeKp(0.1f, 1.0f, 1.0f), makeKp(0.3f, 3.0f, 4.0f), makeKp(0.2f, 2.0f, 1.0f),
            makeKp(0.4f, 0.0f, 1.0f), makeKp(0.4f, 1.0f, 2.0f),};
    CvVector<key_point_t> cvKeyPoints{keyPoints};

    auto start = manager.cv_run(
        Program::TestProgram,
        "squareVector2",
        5,
        true,
        /*image2d_t*/ image, /*npoints*/ 5, /*minBorderX*/ 20, /*minBorderY*/ 20, /*octave*/ 0, /*size*/ 5, cvKeyPoints.kernelArg());

    ASSERT_TRUE(start);

    for (const auto &[lhs, rhs] : boost::combine(keyPoints, cvKeyPoints.result()))
    {
        ASSERT_EQ(5, rhs.class_id);
    }
}


TEST(runKeyPointsKernel, OpenCLTest)
{
    using ORB_SLAM3::opencl::Program;

    auto &manager = ORB_SLAM3::opencl::Manager::the();

    auto mat1      = cv::imread("/home/ego/more/projects/wut-cuda-orb-slam3/datasets/MH01/mav0/cam0/data/"
                                     "1403636579763555584.png", cv::IMREAD_GRAYSCALE);
    mat1.convertTo(mat1, CV_32F, 1.0 / 255);
    auto umat_src  = mat1.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    auto umat_dest = cv::UMat(mat1.size(), cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    auto image = cv::ocl::Image2D(umat_src);
    //    auto cl_buffer = static_cast<cl_mem>(umat_src.handle(cv::ACCESS_RW));

    std::array<key_point_t, 5> keyPoints{
            makeKp(0.1f, 1.0f, 1.0f), makeKp(0.3f, 3.0f, 4.0f), makeKp(0.2f, 2.0f, 1.0f),
            makeKp(0.4f, 0.0f, 1.0f), makeKp(0.4f, 1.0f, 2.0f),
    };

    boost::compute::vector<key_point_t> gpuKeyPoints(5);
    boost::compute::copy(keyPoints.begin(), keyPoints.end(), gpuKeyPoints.begin());

    // __kernel void addBorder_kernel(__global key_point_t *keypoints, int npoints, int minBorderX, int minBorderY, int octave, int size) {
    auto start = manager.cv_run(Program::TestProgram, "squareVector", 5, true, image, gpuKeyPoints.get_buffer().get(),
                             /*npoints*/ 5, /*minBorderX*/ 20, /*minBorderY*/ 20, /*octave*/ 0, /*size*/ 5);

    ASSERT_TRUE(start);

    std::vector<key_point_t> values_out(5);
    boost::compute::copy(gpuKeyPoints.begin(), gpuKeyPoints.end(), values_out.begin());

    for (const auto &[lhs, rhs] : boost::combine(keyPoints, values_out)) {
        ASSERT_EQ(5, rhs.class_id);
    }
}

TEST(runSimpleOpenGLProgram, OpenCLTest)
{
    using ORB_SLAM3::opencl::Program;
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

    auto start = manager.run(Program::TestProgram, "squareVector", 256, gpuValues,
                             /*npoints*/ 5, /*minBorderX*/ 20, /*minBorderY*/ 20, /*octave*/ 0, /*size*/ 5);
    start.wait();

    compute::copy(gpuValues.begin(), gpuValues.end(), values_out.begin());

    for (const auto &[lhs, rhs] : boost::combine(values, values_out)) {
        ASSERT_EQ(lhs.class_id * 5, rhs.class_id);
    }
}
