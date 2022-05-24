#include <fstream>
#include <OpenCL/Manager.hpp>

namespace ORB_SLAM3::opencl {

extern const char *g_angleProgram;

static auto makeContext()
{
    if (!cv::ocl::haveOpenCL()) {
        std::exit(1);
    }

    cv::ocl::Context ctx;
    if (!ctx.create(cv::ocl::Device::TYPE_GPU)) {
        std::exit(1);
    }

    return ctx;
}

cv::ocl::ProgramSource loadFromSourceFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        exit(1);
    }

    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string program(size, ' ');
    std::copy_n(std::istreambuf_iterator(file), size, program.begin());

    return cv::ocl::ProgramSource(program);
}

Manager::Manager() :
    m_context(makeContext()),
    m_device(m_context.device(0)),
    m_queue(m_context, m_device),
    m_workGroupSize(m_device.maxWorkGroupSize()),
    m_programs{
            cv::ocl::Program(loadFromSourceFile("src/OpenCL/Kernel/Test.cl"), "-cl-std=CL2.0", m_errorMsg),
            cv::ocl::Program(loadFromSourceFile("src/OpenCL/Kernel/Angle.cl"), "-cl-std=CL2.0", m_errorMsg),
            cv::ocl::Program(loadFromSourceFile("src/OpenCL/Kernel/Orb.cl"), "-cl-std=CL2.0", m_errorMsg)}
{
}

Manager &Manager::the()
{
    static Manager manager;
    return manager;
}

}// namespace ORB_SLAM3::opencl
