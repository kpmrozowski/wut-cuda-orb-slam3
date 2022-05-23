#include <boost/compute.hpp>
#include <opencv2/core/ocl.hpp>

namespace ORB_SLAM3::opencl {

enum class Program : size_t
{
    TestProgram = 0,
    AddBorder   = 1,
    AngleKernel = 2,
    Count       = 3,
};

class Manager
{
    cv::ocl::Context m_context;
    cv::ocl::Device m_device;
    cv::ocl::Queue m_queue;
    size_t m_workGroupSize;
    std::string m_errorMsg;
    std::array<cv::ocl::Program, static_cast<size_t>(Program::Count)> m_programs;

  public:
    Manager();

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

    template<typename... TArgs>
    auto cv_run(Program programId, const std::string &name, std::size_t count, bool sync, TArgs &&...args)
    {
        cv::ocl::Kernel kernel(name.data(),  cv_program(programId));
        kernel.args(args...);
        return kernel.run(1, &count, &m_workGroupSize, sync, m_queue);
    };

    template<typename... TArgs>
    auto run(Program programId, const std::string &name, std::size_t count, TArgs &&...args)
    {
        boost::compute::kernel kernel(program(programId), name);
        if constexpr (sizeof...(TArgs) > 0) {
            passArgumentsToKernel(0, kernel, args...);
        }
        return queue().enqueue_1d_range_kernel(kernel, 0, count, m_workGroupSize);
    };

    static Manager &the();

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