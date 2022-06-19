#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace std;

int main()
{
    if (!cv::ocl::haveOpenCL())
    {
        cout << "OpenCL is not avaiable..." << endl;
        return 1;
    }
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
        cout << "Failed creating the context..." << endl;
        return 1;
    }

    // In OpenCV 3.0.0 beta, only a single device is detected.
    cout << context.ndevices() << " GPU devices are detected." << endl;
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        cout << "name                 : " << device.name() << endl;
        cout << "available            : " << device.available() << endl;
        cout << "imageSupport         : " << device.imageSupport() << endl;
        cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << endl;
        cout << endl;
    }

    // Select the first device
    cv::ocl::Device(context.device(0));

    // Transfer Mat data to the device
    cv::Mat mat_src = cv::imread("./datasets/MH01/cam0/data/1403636579763555584.png", cv::IMREAD_GRAYSCALE);
    mat_src.convertTo(mat_src, CV_32F, 1.0 / 255);
    cv::UMat umat_src = mat_src.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat umat_dst(mat_src.size(), CV_32F, cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

    std::ifstream ifs("./src/OpenCL/Kernel/shift.cl");
    if (ifs.fail()) return 1;
    std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    cv::ocl::ProgramSource programSource(kernelSource);

    // Compile the kernel code
    cv::String errmsg;
    cv::String buildopt = cv::format("-D dstT=%s -cl-std=CL2.0", cv::ocl::typeToStr(umat_dst.depth())); // "-D dstT=float"
    cv::ocl::Program program = context.getProg(programSource, buildopt, errmsg);

    cv::ocl::Image2D image(umat_src);
    float shift_x = 100.5;
    float shift_y = -50.0;
    cv::ocl::Kernel kernel("shift", program);
    kernel.args(image, shift_x, shift_y, cv::ocl::KernelArg::ReadWrite(umat_dst));

    size_t globalThreads[3] = { static_cast<size_t>(mat_src.cols), static_cast<size_t>(mat_src.rows), 1 };
    //size_t localThreads[3] = { 16, 16, 1 };
    bool success = kernel.run(3, globalThreads, NULL, true);
    if (!success){
        cout << "Failed running the kernel..." << endl;
        return 1;
    }

    // Download the dst data from the device (?)
    cv::Mat mat_dst = umat_dst.getMat(cv::ACCESS_READ);
    std::cout << mat_dst - mat_src << std::endl;

    // cv::imshow("src", mat_src);
    // cv::imshow("dst", mat_dst);
    // cv::waitKey();
    return 0;
}
