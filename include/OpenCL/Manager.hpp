#include <boost/compute.hpp>
#include <algorithm>
#include <opencv2/core/ocl.hpp>
#include <boost/compute.hpp>
#include <concepts>
#include <stdexcept>
#include <string>
#include <span>
#include <utility>

typedef struct
{
    float x;
    float y;
} point2f_t;
typedef struct
{
    point2f_t pt;  //!< coordinates of the keypoints
    float size;    //!< diameter of the meaningful keypoint neighborhood
    float angle;   //!< computed orientation of the keypoint (-1 if not applicable);
    float response;//!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
    int octave;    //!< octave (pyramid layer) from which the keypoint has been extracted
    int class_id;  //!< object class (if the keypoints need to be clustered by an object they belong to)
} key_point_t;

namespace ORB_SLAM3::opencl {

template <class B>
concept Boolean =
  std::movable<std::remove_cvref_t<B>> &&
  requires(const std::remove_reference_t<B>& b1,
           const std::remove_reference_t<B>& b2, const bool a)
{
    requires std::convertible_to<const std::remove_reference_t<B>&, bool>;
    !b1;      requires std::is_convertible_v<decltype(!b1), bool>;
    b1 && a;  requires std::is_same_v<decltype(b1 && a), bool>;
    b1 || a;  requires std::is_same_v<decltype(b1 || a), bool>;
    b1 && b2; requires std::is_same_v<decltype(b1 && b2), bool>;
    a && b2;  requires std::is_same_v<decltype(a && b2), bool>;
    b1 || b2; requires std::is_same_v<decltype(b1 || b2), bool>;
    a || b2;  requires std::is_same_v<decltype(a || b2), bool>;
    b1 == b2; requires std::is_convertible_v<decltype(b1 == b2), bool>;
    b1 == a;  requires std::is_convertible_v<decltype(b1 == a), bool>;
    a == b2;  requires std::is_convertible_v<decltype(a == b2), bool>;
    b1 != b2; requires std::is_convertible_v<decltype(b1 != b2), bool>;
    b1 != a;  requires std::is_convertible_v<decltype(b1 != a), bool>;
    a != b2;  requires std::is_convertible_v<decltype(a != b2), bool>;
};

template<class T>
concept StringLike = std::is_convertible_v<T, std::string_view>;

using namespace std::literals::string_view_literals;
enum class Program : uint8_t
{
    TestProgram = 0,
    AngleKernel = 1,
    AddBorderKernel = 2,
    OrbKernel   = 3,
    TileCalcKeypointsKernel   = 4,
    Count       = 5,
};

template  <typename Key, typename Value, std::size_t Size>
struct Map {
    using Node = std::pair<Key, Value>;
    std::array<Node, Size> data;

    constexpr Value at(const Key &key) const {
        const auto itr = std::find_if(data.begin(), data.end(),
            [&key](const Node &v) { return v.first == key; });
        if (itr != data.end()) {
            return itr->second;
        } else {
            throw std::range_error("Not found");
        }
    }
};

static constexpr auto g_kernels =
    Map<Program, std::string_view, 5>{{
        std::array<std::pair<Program, std::string_view>, 5>{{
            {Program::TestProgram, /*"squareVector"sv, */ "squareVector2"sv},
            {Program::AngleKernel, "IC_Angle_kernel"sv},
            {Program::AddBorderKernel, "addBorder_kernel"sv},
            {Program::OrbKernel, "calcOrb_kernel"sv},
            {Program::TileCalcKeypointsKernel, "tileCalcKeypoints_kernel"sv},
}}}};

class SyncBool
{
    bool m_sync = true;

  public:
    SyncBool() = delete;
    SyncBool(bool b) : m_sync(b) {}
    constexpr bool sync() noexcept { return m_sync; }
};

class Manager
{
    cv::ocl::Context m_context;
    cv::ocl::Device m_device;
    cv::ocl::Queue m_queue;
    std::vector<size_t> m_workGroupSize;
    std::map<Program, std::string> m_errorMsg;
    std::array<cv::ocl::Program, static_cast<size_t>(Program::Count)> m_programs;

  public:
    Manager();

    static Manager &the();

    [[nodiscard]] constexpr cv::ocl::Context &cv_context()
    {
        return m_context;
    }

    [[nodiscard]] boost::compute::device device()
    {
        return boost::compute::device{static_cast<cl_device_id>(m_device.ptr())};
    }

    [[nodiscard]] constexpr cv::ocl::Device &cv_device()
    {
        return m_device;
    }

    [[nodiscard]] boost::compute::command_queue queue()
    {
        return boost::compute::command_queue{static_cast<cl_command_queue>(m_queue.ptr())};
    }

    [[nodiscard]] constexpr cv::ocl::Queue &cv_queue()
    {
        return m_queue;
    }

    [[nodiscard]] boost::compute::program program(Program program)
    {
        return boost::compute::program{
                static_cast<cl_program>(cv_program(program).ptr())};
    }

    [[nodiscard]] constexpr cv::ocl::Program &cv_program(Program program)
    {
        return m_programs[static_cast<size_t>(program)];
    }

    template<size_t DIMS, typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0, StringLike S, typename... TArgs>
    auto cv_run(Program programId, const S &name, size_t *global_work_size, T syncBool, TArgs &&...args)
    {
        std::string kernelName{name};
        cv::ocl::Kernel kernel(kernelName.data(),  cv_program(programId));
        kernel.args(args...);
        return kernel.run(DIMS , global_work_size, m_workGroupSize.data(), syncBool, m_queue);
    };

    template<size_t DIMS, typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0, typename... TArgs>
    auto cv_run(Program programId, size_t *global_work_size, T syncBool, TArgs &&...args)
    {
        cv::ocl::Kernel kernel(g_kernels.at(programId).data(),  cv_program(programId));
        kernel.args(args...);
        return kernel.run(DIMS , global_work_size, m_workGroupSize.data(), syncBool, m_queue);
    };

    template<size_t DIMS, typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0, StringLike S, typename... TArgs>
    auto cv_run(Program programId, const S &name, size_t *global_work_size, size_t *local_work_group, T syncBool, TArgs &&...args)
    {
        std::string kernelName{name};
        cv::ocl::Kernel kernel(kernelName.data(),  cv_program(programId));
        kernel.args(args...);
        return kernel.run(DIMS , global_work_size, local_work_group, syncBool, m_queue);
    };

    template<size_t DIMS, typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0, typename... TArgs>
    auto cv_run(Program programId, size_t *global_work_size, size_t *local_work_group, T syncBool, TArgs &&...args)
    {
        cv::ocl::Kernel kernel(g_kernels.at(programId).data(),  cv_program(programId));
        kernel.args(args...);
        return kernel.run(DIMS , global_work_size, local_work_group, syncBool, m_queue);
    };

    template<typename... TArgs>
    auto run(Program programId, const std::string &name, std::size_t global_work_size, TArgs &&...args)
    {
        boost::compute::kernel kernel(program(programId), name);
        if constexpr (sizeof...(TArgs) > 0) {
            passArgumentsToKernel(0, kernel, args...);
        }
        //        queue().enqueue_native_kernel()
        return queue().enqueue_1d_range_kernel(kernel, 0, global_work_size, m_workGroupSize[0]);
    }

    template<typename... TArgs>
    auto run2(Program programId, const std::string &name, std::size_t global_work_size, std::size_t  local_work_group, TArgs &&...args)
    {
        boost::compute::kernel kernel(program(programId), name);
        if constexpr (sizeof...(TArgs) > 0) {
            passArgumentsToKernel(0, kernel, args...);
        }
        //        queue().enqueue_native_kernel()
        return queue().enqueue_1d_range_kernel(kernel, 0, global_work_size, local_work_group);
    }

  private:
    template<typename TFirst, typename... TOthers>
    constexpr void passArgumentsToKernel(std::size_t index, boost::compute::kernel &kernel, TFirst &&first,
                                         TOthers &&...others)
    {
        kernel.set_arg(index, std::forward<TFirst>(first));
        if constexpr (sizeof...(TOthers) > 0) {
            passArgumentsToKernel(index + 1, kernel, others...);
        }
    }
};

}// namespace ORB_SLAM3::opencl

BOOST_COMPUTE_ADAPT_STRUCT(point2f_t, point2f_t, (x, y))


BOOST_COMPUTE_ADAPT_STRUCT(key_point_t, key_point_t, (pt, size, angle, response, octave, class_id))
