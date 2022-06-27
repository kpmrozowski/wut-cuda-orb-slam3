#include "OpenCL/Manager.hpp"
#include <algorithm>
#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/compute.hpp>
#include <boost/compute/types/struct.hpp>
#include <boost/range/algorithm_ext/for_each.hpp>
#include <boost/range/combine.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/core/mat.hpp>
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
    boost::compute::device device = boost::compute::system::default_device();
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
    for (int i = 0; i < context.ndevices(); i++) {
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
    bool synchronized            = true;

  public:
    explicit CvVector(std::vector<T> &&vec) :
        m_vec(vec),
        m_size(m_vec.size()),
        m_mat_before(1, m_vec.size() * sizeof(T), CV_TYPE, m_vec.data())
    {
    }

    [[nodiscard]] constexpr size_t size() { return m_size; }

    [[nodiscard]] std::vector<T> &modify() { return m_vec; }

    [[nodiscard]] std::span<T> before()
    {
        return std::span<T>{reinterpret_cast<T *>(m_mat_before.data), m_size};
    }

    [[nodiscard]] cv::UMat &umat()
    {
        synchronized = false;
        m_umat       = m_mat_before.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        return m_umat;
    }

    [[nodiscard]] cv::ocl::KernelArg kernelArg()
    {
        synchronized = false;
        m_umat       = m_mat_before.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        return cv::ocl::KernelArg::ReadWrite(m_umat);
    }

    std::span<T> result()
    {
        if (not m_mat.has_value()) {
            m_mat = m_umat.getMat(cv::ACCESS_READ);
            return std::span<T>{reinterpret_cast<T *>(m_mat_before.data), m_size};
        }
        return {reinterpret_cast<T *>(m_mat.value().data), m_size};
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

cv::Mat &load_sample_image()
{
    // auto filename = "./datasets/MH01/mav0/cam0/data/1403636579763555584.png";
    auto filename =
            "/home/ego/more/projects/wut-cuda-orb-slam3/datasets/dataset/sequences/03/image_0/000000.png";
    if (not std::filesystem::is_regular_file(filename)) {
        throw std::runtime_error("image does not exist");
    } else {
        std::cout << "image exists\n";
    }
    static cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    // image.convertTo(image, CV_32F, 1.0 / 255);
    return image;
}

TEST(runKeyPointsKernel2, OpenCLTest)
{
    auto &manager      = ORB_SLAM3::opencl::Manager::the();
    cv::Mat &image     = load_sample_image();
    cv::UMat umat_src  = image.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat umat_dest = cv::UMat(image.size(), cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::ocl::Image2D image2d{umat_src};

    CvVector<key_point_t> cvKeyPoints{
            {
             makeKp(0.1f, 1.0f, 1.0f),
             makeKp(0.3f, 3.0f, 4.0f),
             makeKp(0.2f, 2.0f, 1.0f),
             makeKp(0.4f, 0.0f, 1.0f),
             makeKp(0.4f, 1.0f, 2.0f),
             }
    };

    auto npoints = cvKeyPoints.size();

    std::vector<size_t> globalDim{npoints};
    auto start = manager.cv_run<1>(Program::TestProgram, "squareVector2", globalDim.data(), true,
                                   /*image2d_t*/ image2d,
                                   /*npoints*/ 5,
                                   /*minBorderX*/ 20,
                                   /*minBorderY*/ 20,
                                   /*octave*/ 0,
                                   /*size*/ 5,
                                   /*keypoints*/ cvKeyPoints.kernelArg());

    ASSERT_TRUE(start);

    for (const auto &[lhs, rhs] : boost::combine(cvKeyPoints.before(), cvKeyPoints.result())) {
        ASSERT_EQ(5, rhs.class_id);
        std::cout << 5 << " == " << rhs.class_id << "\n";
    }
}

// BOOST COMPUTE IS NOT WORKING AT THE MOMENT
// TEST(runKeyPointsKernel, OpenCLTest)
// {
//     auto &manager = ORB_SLAM3::opencl::Manager::the();

//     cv::Mat& image = load_sample_image();
//     cv::UMat umat_src = image.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
//     auto umat_dest = cv::UMat(image.size(), cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
//     cv::ocl::Image2D image2d{umat_src};
//     //    auto cl_buffer = static_cast<cl_mem>(umat_src.handle(cv::ACCESS_RW));

//     std::array<key_point_t, 5> keyPoints{
//             makeKp(0.1f, 1.0f, 1.0f), makeKp(0.3f, 3.0f, 4.0f), makeKp(0.2f, 2.0f, 1.0f),
//             makeKp(0.4f, 0.0f, 1.0f), makeKp(0.4f, 1.0f, 2.0f),
//     };

//     boost::compute::vector<key_point_t> gpuKeyPoints(5);
//     boost::compute::copy(keyPoints.begin(), keyPoints.end(), gpuKeyPoints.begin());

//     // __kernel void addBorder_kernel(__global key_point_t *keypoints, int npoints, int minBorderX, int minBorderY, int octave, int size) {
//     std::vector<size_t> gridDim{5};
//     auto start = manager.cv_run<1>(
//         Program::TestProgram,
//         "squareVector",
//         gridDim.data(),
//         true,
//         image2d,
//         gpuKeyPoints.get_buffer().get(),
//         /*npoints*/ 5,
//         /*minBorderX*/ 20,
//         /*minBorderY*/ 20,
//         /*octave*/ 0,
//         /*size*/ 5
//     );

//     ASSERT_TRUE(start);

//     std::vector<key_point_t> values_out(5);
//     boost::compute::copy(gpuKeyPoints.begin(), gpuKeyPoints.end(), values_out.begin());

//     for (const auto &[lhs, rhs] : boost::combine(keyPoints, values_out)) {
//         ASSERT_EQ(5, rhs.class_id);
//     }
// }

//TEST(runSimpleOpenGLProgram, OpenCLTest)
//{
//    namespace compute = boost::compute;
//
//    //    std::string source = BOOST_COMPUTE_STRINGIZE_SOURCE(
//    //        __kernel void squareVector(__global int *data) {
//    //            const int globalId   = get_global_id(0);
//    //            const int value = data[globalId];
//    //            data[globalId] = value * value;
//    //        }
//    //    );
//
//    auto &manager = ORB_SLAM3::opencl::Manager::the();
//
//    std::vector<key_point_t> values(256), values_out(256);
//    std::for_each(values.begin(), values.end(), [i = 0](key_point_t &v) mutable { v.class_id = ++i; });
//    //    std::iota(values.begin(), values.end(), 0);
//
//    compute::vector<key_point_t> gpuValues(256);
//    compute::copy(values.begin(), values.end(), gpuValues.begin());
//
//    auto start = manager.run(Program::TestProgram, "squareVector", 256, gpuValues,
//                             /*npoints*/ 5,
//                             /*minBorderX*/ 20,
//                             /*minBorderY*/ 20,
//                             /*octave*/ 0,
//                             /*size*/ 5);
//    start.wait();
//
//    compute::copy(gpuValues.begin(), gpuValues.end(), values_out.begin());
//
//    for (const auto &[lhs, rhs] : boost::combine(values, values_out)) {
//        ASSERT_EQ(lhs.class_id * 5, rhs.class_id);
//    }
//}

TEST(runCalcOrbKernel, OpenCLTest)
{
    auto &manager  = ORB_SLAM3::opencl::Manager::the();
    cv::Mat &image = load_sample_image();
    image.convertTo(image, CV_32F, 1.0 / 255);
    cv::UMat umat_src = image.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::ocl::Image2D image2d{umat_src};

    CvVector<key_point_t> cvKeyPoints{
            {
             makeKp(0.1f, 1.0f, 1.0f),
             makeKp(0.3f, 3.0f, 4.0f),
             makeKp(0.2f, 2.0f, 1.0f),
             makeKp(0.4f, 0.0f, 1.0f),
             makeKp(0.4f, 1.0f, 2.0f),
             }
    };

    auto npoints = cvKeyPoints.size();

    CvVector cvDescriptors{std::vector<uint>(cvKeyPoints.size() * 32)};

    std::vector<size_t> blockDim{32};
    std::vector<size_t> globalDim{blockDim[0] * npoints};
    // std::vector<size_t> blockDim{256};
    // std::vector<size_t> globalDim{blockDim[0] * cv::divUp(npoints, blockDim[0])};
    auto start = manager.cv_run<1>(Program::OrbKernel, globalDim.data(), blockDim.data(), true,
                                   /*image2d_t */ image2d,
                                   /*char* */ cvKeyPoints.umat(),
                                   /*image2d_t */ cvDescriptors.umat());

    ASSERT_TRUE(start);

    // std::cout << "cvDescriptors = [\n";
    // for (int32_t i = 0; i < cvKeyPoints.size(); ++i)
    // {
    //     for (int32_t j = 0; j < 32; ++j)
    //         std::cout << cvDescriptors.result()[j + 32 * i] << ", ";
    //     std::cout << "\n";
    // }
    // std::cout << "]\n";
}

struct short2
{
    short x;
    short y;
};

/*
################### CUDA SETTINGS: ###################
dimGrid: 38, 11, 1
dimBlock: 32, 8, 1
image: 343, 1210,
image.type: 0
maxKeypoints: 10000
Threshold: 20, 7
scoreUMat: 343, 1210

################### CUDA BEFORE: ###################
kpLoc: 0, 0, 0, 0
kpScore: 0, 0
counterPtr: 0

################### CUDA AFTER: ###################
kpLoc: 191, 3, 191, 10
kpScore: 7, 9
counterPtr: 5927

CUDA ExtractORB duration: 0.00975116

################### OpenCL SETTINGS: ###################
dimGrid: 39, 12
dimBlock: 32, 8
scoreMat: 375, 1242
image: 375, 1242,
image.type: 0
maxKeypoints: 10000
Threshold: 20, 7

################### OpenCL BEFORE: ###################
kpLoc: (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
kpScore: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
counter_ptr: 0

running kernel
kernel finished

################### OpenCL AFTER: ###################
kpLoc: (29, 24), (23, 29), (23, 22), (26, 22), (22, 25), (192, 192), (187, 194), (29, 24), (31, 21), (188, 207), (186, 219), (31, 28), (192, 201), (22, 25), (190, 213), (23, 22), (26, 22), (23, 29), (189, 196), (176, 217),
kpScore: 63, 56, 107, 87, 34, 23, 26, 63, 12, 23, 22, 8, 42, 34, 50, 107, 87, 56, 36, 46,
counter_ptr: 340

OpenCL duration: 0.00044828


Mat::type's
 	    C1 	C2 	C3 	C4 	C5  C6 	C7 	C8
CV_8U 	0 	8 	16 	24 	32 	40 	48 	56
CV_8S 	1 	9 	17 	25 	33 	41 	49 	57
CV_16U 	2 	10 	18 	26 	34 	42 	50 	58
CV_16S 	3 	11 	19 	27 	35 	43 	51 	59
CV_32S 	4 	12 	20 	28 	36 	44 	52 	60
CV_32F 	5 	13 	21 	29 	37 	45 	53 	61
CV_64F 	6 	14 	22 	30 	38 	46 	54 	62
 */

using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;

std::tuple<bool, std::vector<short2>> runTileCalcKeypointsKernel_fun(cv::Mat &image)
{
    auto &manager     = ORB_SLAM3::opencl::Manager::the();
    cv::UMat umat_src = image.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::ocl::Image2D image2d{umat_src};
    uint maxKeypoints = 10'000;

    // struct { uint x = 39; uint y = 12; } dimGrid;
    // struct { uint x = 32; uint y = 8; } dimBlock;
    const struct
    {
        uint x = 32;
        uint y = 8;
    } dimBlock;

    const struct
    {
        uint x;
        uint y;
    } dimGrid = {(uint) cv::divUp(image.cols, dimBlock.x), (uint) cv::divUp(image.rows, dimBlock.y * 4u)};

    CvVector kpLoc{std::vector<short2>(maxKeypoints)};
    CvVector kpScore{std::vector<float>(maxKeypoints)};
    uint highThreshold = 20, lowThreshold = 7;
//    cv::Mat scoreMat{image.size(), CV_8S, cv::USAGE_ALLOCATE_DEVICE_MEMORY};
//    auto scoreUMat = scoreMat.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
//    CvVector scoreUMat{std::vector<int>(image.rows * image.cols)};
    CvVector counterPtr{std::vector<uint>(1)};
    CvVector debugMat{std::vector<uint>(dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y)};
    CvVector scoreUMat{std::vector<int>(image.rows * image.cols * 4)};

    std::cout << "\n################### SETTINGS: ###################"
              << "\ndimGrid: " << dimGrid.x << ", " << dimGrid.y << "\ndimBlock: " << dimBlock.x << ", "
//              << dimBlock.y << "\nscoreMat: " << scoreUMat.rows << ", " << scoreUMat.cols
              << "\nimage: " << image.rows << ", " << image.cols << ", "
              << "\nimage: " << (image.rows * image.cols) << '\n'
              << "\ngrid: " << (dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y) << '\n'
              << "\nimage.type: " << image.type() << "\nmaxKeypoints: " << maxKeypoints
              << "\nThreshold: " << highThreshold << ", " << lowThreshold <<

            "\n\n################### BEFORE: ###################"
              << "\nkpLoc: ";
    for (int i = 0; i < 20; ++i) {
        std::cout << "(" << kpLoc.before()[i].x << ", " << kpLoc.before()[i].y << "), ";
    }
    std::cout << "\nkpScore: ";
    for (int i = 0; i < 20; ++i) {
        std::cout << kpScore.before()[i] << ", ";
    }
    std::cout << "\ncounter_ptr: " << counterPtr.before()[0] << std::endl;

    std::vector<size_t> globalDim{dimGrid.x * dimBlock.x, dimGrid.y * dimBlock.y};
    std::vector<size_t> blockDim{dimBlock.x, dimBlock.y};
    std::cout << "\nrunning kernel";
    static chrono_tp startTime = std::chrono::high_resolution_clock::now();
    auto start = manager.cv_run<2>(Program::TileCalcKeypointsKernel, globalDim.data(), blockDim.data(), true,
                                   /*image2d_t */ image2d,
                                   /*short2* */ kpLoc.kernelArg(),
                                   /*float* */ kpScore.kernelArg(),
                                   /*uint */ maxKeypoints,
                                   /*uint */ highThreshold,
                                   /*uint */ lowThreshold,
                                   /*int* */ scoreUMat.kernelArg(),
                                   /*uint* */ debugMat.kernelArg(),
                                   /*uint* */ counterPtr.kernelArg(),
                                   image.rows,
                                   image.cols);
    chrono_tp currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    std::cout << "\nkernel finished\n";
    if (not start) {
        return {start, {}};
    }
    auto kpLocResult   = kpLoc.result();
    auto kpScoreResult = kpScore.result();
    std::vector<short2> kpLocVec{kpLocResult.begin(), kpLocResult.end()};
    std::vector<float> kpScoreVec{kpScoreResult.begin(), kpScoreResult.end()};
    kpLocVec.resize(counterPtr.result()[0]);
    kpScoreVec.resize(counterPtr.result()[0]);
    // std::cout <<
    //     "\n################### AFTER: ###################" <<
    //     "\nkpLoc: ";
    // for (int i = 0; i < kpLocVec.size(); ++i) {
    //     std::cout << "(" << kpLocVec[i].x << ", " << kpLocVec[i].y << ")" << ", ";
    // }
    // std::cout << "\nkpScore: ";
    // for (int i = 0; i < kpScoreVec.size(); ++i) {
    //     std::cout << kpScoreVec[i] << ", ";
    // }
    // std::cout <<
    //     "\ncounter_ptr: " << counterPtr.result()[0] <<
    //     "\nduration: " << time << std::endl;
    // auto scoreMat = scoreUMat.getMat(cv::ACCESS_READ);

    // for (int blockIdxY = 0; blockIdxY < dimGrid.y; ++blockIdxY) {
    //     for (int threadIdxY = 0; threadIdxY < dimBlock.y; ++threadIdxY) {
    //         for (int blockIdxX = 0; blockIdxX < dimGrid.x; ++blockIdxX) {
    //             for (int threadIdxX = 0; threadIdxX < dimBlock.x; ++threadIdxX) {
    //                 std::cout << scoreMat.at<int>(threadIdxY + blockIdxY * dimBlock.y, threadIdxX + blockIdxX * dimBlock.x) << ",";
    //             }
    //         }
    //         std::cout << "\n";
    //     }
    // }
    // std::cout << "\ndimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y = "
    //     << dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y << "\n";
    // std::cout << "\nglobalDim[0] * globalDim[1] = "
    //     << globalDim[0] * globalDim[1] << "\n";
    return {start, kpLocVec};
}

TEST(runTileCalcKeypointsKernel, OpenCLTest)
{
    cv::Mat &image = load_sample_image();
    ASSERT_TRUE(image.type() == CV_8UC1);
    auto [start, kpLoc] = runTileCalcKeypointsKernel_fun(image);
    ASSERT_TRUE(start);
}

/*
################### CUDA SETTINGS: ###################
dimGrid: 55, 1, 1
dimBlock: 32, 8, 1
image: 375, 1242,
image.type: 0
npoints: 436
half_k: 15

################### CUDA BEFORE: ###################
kpLoc: -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,

################### CUDA AFTER: ###################
kpLoc: 316.081, 187.157, 87.8479, 25.6011, 60.0616, 275.594, 15.7805, 307.736, 277.272, 350.468, 23.8067, 34.1896, 275.354, 202.223, 232.828, 263.79, 299.762, 190.925, 224.431, 153.109,

CUDA IC_Angle_kernel duration: 0.000138659
 */
TEST(runICAngleKernel, OpenCLTest)
{
    auto &manager  = ORB_SLAM3::opencl::Manager::the();
    cv::Mat &image = load_sample_image();
    // cv::imshow("", image);
    // cv::waitKey();
    ASSERT_TRUE(image.type() == CV_8UC1);
    cv::UMat umat_src = image.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

    auto [start1, kpLoc] = runTileCalcKeypointsKernel_fun(image);
    ASSERT_TRUE(start1);

    uint npoints = kpLoc.size();
    uint half_k  = 15u;

    CvVector cvKeyPoints{std::vector<key_point_t>(npoints)};
    CvVector cvAngles{std::vector<float>(npoints)};
    boost::for_each(cvKeyPoints.modify(), kpLoc, [](key_point_t &kp, short2 loc) {
        kp.pt.x  = loc.x;
        kp.pt.y  = loc.y;
        kp.angle = -1.f;
    });

    // struct { uint x = 55; uint y = 1; } dimGrid;
    // struct { uint x = 32; uint y = 8; } dimBlock;
    const struct
    {
        uint x = 32;
        uint y = 8;
    } dimBlock;

    const struct
    {
        uint x;
        uint y;
    } dimGrid = {(uint) cv::divUp(static_cast<int>(npoints), (dimBlock.y)), 1u};

    CvVector debugMat{std::vector<uint>(dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y)};

    // std::cout <<
    //     "\n################### SETTINGS: ###################\n" <<
    //     "\ndimGrid: " << dimGrid.x << ", " << dimGrid.y << ", " <<
    //     "\ndimBlock: " << dimBlock.x << ", " << dimBlock.y << ", " <<
    //     "\nimage: " << image.rows << ", " << image.cols << ", " <<
    //     "\nimage.type: " << image.type() <<
    //     "\nnpoints: " << npoints <<
    //     "\nhalf_k: " << half_k <<

    //     "\n\n################### BEFORE: ###################\n" <<
    //     "\nkp.angle: ";
    // for (int i = 0; i < 20; ++i) {
    //     std::cout << cvKeyPoints.before()[i].angle <<", ";
    // }
    // std::cout << std::endl;

    std::vector<size_t> globalDim{dimGrid.x * dimBlock.x, dimGrid.y * dimBlock.y};
    std::vector<size_t> blockDim{dimBlock.x, dimBlock.y};
    std::cout << "\nrunning kernel";
    static chrono_tp startTime = std::chrono::high_resolution_clock::now();
    auto start            = manager.cv_run<2>(Program::AngleKernel, globalDim.data(), blockDim.data(), true,
                                   /*unsigned char* */ umat_src,
                                   /*key_point_t* */ cvKeyPoints.kernelArg(),
                                   /*uint */ npoints,
                                   /*uint */ half_k,
                                   /*uint* */ debugMat.kernelArg());
    chrono_tp currentTime = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(start);
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    // std::cout << "\nkernel finished\n";
    // std::cout <<
    //     "\n################### AFTER: ###################\n" <<
    //     "\nkp.angle: ";
    // for (int i = 0; i < 20; ++i) {
    //     std::cout << cvKeyPoints.result()[i].angle <<", ";
    // }
    // std::cout << "\n\n";
    // auto debugMatResult = debugMat.result();
    // for (int blockIdxY = 0; blockIdxY < dimGrid.y; ++blockIdxY) {
    //     for (int threadIdxY = 0; threadIdxY < dimBlock.y; ++threadIdxY) {
    //         for (int blockIdxX = 0; blockIdxX < dimGrid.x; ++blockIdxX) {
    //             for (int threadIdxX = 0; threadIdxX < dimBlock.x; ++threadIdxX) {
    //                 std::cout << debugMatResult[dimBlock.x * dimGrid.x * (threadIdxY + blockIdxY * dimBlock.y) + threadIdxX + blockIdxX * dimBlock.x] << ",";
    //             }
    //         }
    //         std::cout << "\n";
    //     }
    // }
}
