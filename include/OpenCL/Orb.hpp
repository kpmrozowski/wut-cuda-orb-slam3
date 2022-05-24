#pragma once
#ifndef __ORB_HPP__
#define __ORB_HPP__

#include <vector>
#include <opencv2/core/ocl.hpp>
//#include <opencv2/core/opencl/ocl_defs.hpp>
//#include <clc/clc.h>
#include <boost/compute.hpp>

using GpuKeyPoints = boost::compute::vector<cv::KeyPoint>;

constexpr auto g_openCLProgram = BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void squareVector(__global int *values) {
                const int globalId = get_global_id(0); // groupId * size  + local
                const int groupId = get_group_id(0); // upto 8
                const int localId = get_local_id(0); // upto 256
                const int value = values[globalId];
                values[globalId] = value * groupId;
        }
);

namespace ORB_SLAM3::opencl {

class GpuOrb {
    unsigned int m_maxKeyPoints;
    GpuKeyPoints m_gpuKeyPoints;
    cv::UMat m_gpuDescriptors;
    cv::UMat m_gpuDesc;
    cv::ocl::Queue m_queue;
  public:
    GpuOrb(int maxKeyPoints = 10000);
    ~GpuOrb();

    void launch_async(cv::InputArray _image, const cv::KeyPoint * _keyPoints, const int nPoints);
    void join(cv::Mat &_descriptors);

    static void loadPattern(const cv::Point * _pattern);
};

}
#endif
