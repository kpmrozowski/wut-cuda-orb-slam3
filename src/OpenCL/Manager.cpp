#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <OpenCL/Manager.hpp>
#include <string>

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
        std::cerr << "no such file or directory: " << filename << '\n';
        exit(1);
    }
    std::ifstream ifs(filename);
    std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

//    file.seekg(0, std::ios::end);
//    auto size = file.tellg();
//    file.seekg(0, std::ios::beg);
//
//    std::string program(size, ' ');
//    std::copy_n(std::istreambuf_iterator(file), size, program.begin());

    return cv::ocl::ProgramSource(kernelSource);
}

Benchmark::Benchmark() :
    m_context(makeContext()),
    m_device(m_context.device(0)),
    m_queue(m_context, m_device),
    m_workGroupSize({m_device.maxWorkGroupSize(), 1, 1}),
    m_programs{
            m_context.getProg(loadFromSourceFile("src/OpenCL/Kernel/Test.cl"), "-cl-std=CL2.0 -DRETURN_STATUSES", m_errorMsg[Program::TestProgram]),
            m_context.getProg(loadFromSourceFile("src/OpenCL/Kernel/Angle.cl"), "-cl-std=CL2.0 -DRETURN_STATUSES", m_errorMsg[Program::AngleKernel]),
            m_context.getProg(loadFromSourceFile("src/OpenCL/Kernel/AddBorder.cl"), "-cl-std=CL2.0 -DRETURN_STATUSES", m_errorMsg[Program::AddBorderKernel]),
            m_context.getProg(loadFromSourceFile("src/OpenCL/Kernel/Orb.cl"), "-cl-std=CL2.0 -DRETURN_STATUSES", m_errorMsg[Program::OrbKernel]),
            m_context.getProg(loadFromSourceFile("src/OpenCL/Kernel/TileCalcKeypoints.cl"), "-cl-std=CL2.0 -DRETURN_STATUSES", m_errorMsg[Program::TileCalcKeypointsKernel])
    }
{
    bool success = true;
    int failedCount = 0;
    std::for_each(m_errorMsg.begin(), m_errorMsg.end(), [&success, &failedCount](std::pair<Program, std::string> &&p) {
        if (not p.second.empty()) {
            ++failedCount;
            success = false;
            std::cout << "#############################################################################\n"
                "\n>>>>>> Kernel \"" << g_kernels.at(p.first) << "\" failed to compile due to below error <<<<<<\n\n" <<
                p.second <<
                "\n>>>>>> Kernel \"" << g_kernels.at(p.first) << "\" failed to compile due to above error <<<<<<\n\n"
                "#############################################################################\n\n";
        }
    });
    std::cout << "Kernels that failed to compile (quantity=" << failedCount << "): ";
    std::for_each(m_errorMsg.begin(), m_errorMsg.end(), [&success, &failedCount](std::pair<Program, std::string> &&p) {
        if (not p.second.empty()) {
            success = false;
            std::cout << g_kernels.at(p.first);
            if (--failedCount) {
                std::cout << ", ";
            } else {
                std::cout << ".\nExiting...";
            }
        }
    });
    std::cout << "\n";
    if (not success) {
        exit(1);
    }
}

Benchmark &Benchmark::the()
{
    static Benchmark manager;
    return manager;
}

}// namespace ORB_SLAM3::opencl
