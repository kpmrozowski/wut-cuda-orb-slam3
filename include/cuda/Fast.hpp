#pragma once
#ifndef __FAST_HPP__
#define __FAST_HPP__

#include <vector>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <cuda/Cuda.hpp>

namespace ORB_SLAM3 { namespace cuda {
//   using namespace std;
//   using namespace cv;
//   using namespace cv::cuda;

  const float FEATURE_SIZE = 7.0;

  class GpuFast {
    short2 * kpLoc;
    float * kpScore;
    unsigned int * counter_ptr;
    unsigned int highThreshold;
    unsigned int lowThreshold;
    unsigned int maxKeypoints;
    unsigned int count;
    cv::cuda::GpuMat scoreMat;
    cudaStream_t stream;
    cv::cuda::Stream cvStream;
  public:
    GpuFast(int highThreshold, int lowThreshold, int maxKeypoints = 10000);
    ~GpuFast();

    void detect(cv::InputArray, std::vector<cv::KeyPoint>&);

    void detectAsync(cv::InputArray);
    void joinDetectAsync(std::vector<cv::KeyPoint>&);
  };

  class IC_Angle {
    unsigned int maxKeypoints;
    cv::KeyPoint * keypoints;
    cudaStream_t stream;
    cv::cuda::Stream _cvStream;
  public:
    IC_Angle(unsigned int maxKeypoints = 10000);
    ~IC_Angle();
    void launch_async(cv::InputArray _image, cv::KeyPoint * _keypoints, int npoints, int half_k, int minBorderX, int minBorderY, int octave, int size);
    void join(cv::KeyPoint * _keypoints, int npoints);

    cv::cuda::Stream& cvStream() { return _cvStream;}
    static void loadUMax(const int* u_max, int count);
  };
} }
#endif
