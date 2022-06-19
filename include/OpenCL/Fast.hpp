#pragma once
#ifndef __FAST_HPP__
#define __FAST_HPP__

#include <CL/cl.h>
#include <opencv2/core/ocl.hpp>
#include <vector>

namespace ORB_SLAM3::opencl {

const float FEATURE_SIZE = 7.0;

class GpuFast
{
    cl_short2 *kpLoc;
    float *kpScore;
    unsigned int *counter_ptr;
    unsigned int highThreshold;
    unsigned int lowThreshold;
    unsigned int maxKeyPoints;
    unsigned int count;
    cv::UMat scoreMat;
    cv::ocl::Queue queue;

  public:
    GpuFast(int highThreshold, int lowThreshold, int maxKeyPoints = 10000);
    ~GpuFast();

    void detect(cv::InputArray, std::vector<cv::KeyPoint> &);

    void detectAsync(cv::InputArray);
    void joinDetectAsync(std::vector<cv::KeyPoint> &);
};

class IC_Angle
{
    unsigned int maxKeyPoints;
    cv::KeyPoint *keyPoints;
    cv::ocl::Queue m_queue;

  public:
    IC_Angle(unsigned int maxKeyPoints = 10000);
    ~IC_Angle();
    void launch_async(cv::InputArray _image, cv::KeyPoint *_keyPoints, int nPoints, int half_k,
                      int minBorderX, int minBorderY, int octave, int size);
    void join(cv::KeyPoint *_keyPoints, int nPoints);

    cv::ocl::Queue &queue() { return m_queue; }

    static void loadUMax(const int *u_max, int count);
};
}// namespace ORB_SLAM3::opencl
#endif
