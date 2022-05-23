#include <boost/compute.hpp>
#include <fmt/core.h>
#include <iostream>
#include <OpenCL/Manager.hpp>
#define PR(x) fmt::print("{}\n", x);

struct Test
{
    int foo;
    int bar;
};

void runSimpleOpenGLProgram()
{
    using ORB_SLAM3::opencl::Program;
    namespace compute = boost::compute;

    auto &manager = ORB_SLAM3::opencl::Manager::the();

    const auto work_group_size = manager.device().max_work_group_size();

    std::vector<Test> values(2048);
    std::generate(values.begin(), values.end(), [a = 0, b = 1]() mutable -> Test {
        auto c = a;
        return {a = b, b = b + c};
    });

    std::cout << "before: \n";
    for (auto value : values) {
        std::cout << value.foo << " " << value.bar << '\n';
    }

    compute::vector<Test> gpuValues(2048);
    compute::copy(values.begin(), values.end(), gpuValues.begin());

    auto start = manager.run(Program::TestProgram, "squareVector",  2048, gpuValues);

//    compute::kernel kernel(manager.program(ORB_SLAM3::opencl::Program::TestProgram), "squareVector");
//    kernel.set_arg(0, gpuValues);
//
//    auto start = manager.queue().enqueue_1d_range_kernel(kernel, 0, 2048, work_group_size);
    start.wait();

    compute::copy(gpuValues.begin(), gpuValues.end(), values.begin());

    std::cout << "after: \n";
    for (auto value : values) {
        std::cout << value.foo << " " << value.bar << '\n';
    }
}

int main() { runSimpleOpenGLProgram(); }