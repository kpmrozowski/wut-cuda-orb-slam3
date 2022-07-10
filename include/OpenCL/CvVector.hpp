#include <opencv2/core/mat.hpp>
#include <opencv2/core/ocl.hpp>
#include <optional>
#include <span>

namespace ORB_SLAM3::opencl {

struct uint2 {
    uint x, y;
};

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
        m_vec(std::move(vec)),
        m_mat_before(1, static_cast<int>(m_vec.size() * sizeof(T)), CV_TYPE, m_vec.data()),
        m_size(m_vec.size())
    {
    }
    // explicit CvVector(T* vec, size_t size) :
    //     m_vec(vec, vec + size),
    //     m_size(size),
    //     m_mat_before(1, m_size * sizeof(T), CV_TYPE, m_vec.data())
    // {
    // }
    explicit CvVector(T* vecData, size_t size) :
            m_size(size),
            m_mat_before(1, m_size * sizeof(T), CV_TYPE, vecData)
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

    T* resultPtr()
    {
        m_mat = cv::Mat{1, static_cast<int>(m_vec.size() * sizeof(T)), CV_TYPE};
        m_mat = m_umat.getMat(cv::ACCESS_READ);
        return reinterpret_cast<T*>(m_mat.value().data);
    }
};

// template<typename T, int CV_TYPE = CV_8SC1>
// class CvVector
// {
//     std::vector<T> m_vec;
//     size_t m_size;
//     cv::UMat m_umat;
//     cv::Mat m_mat;
//     bool synchronized            = true;

//   public:
//     explicit CvVector(std::vector<T> &&vec) :
//             m_vec(std::move(vec)),
//             m_size(m_vec.size()),
//             m_mat(1, m_size * sizeof(T), CV_TYPE, m_vec.data())
//     {
//     }
//     explicit CvVector(std::vector<T> &vec) :
//             m_size(vec.size()),
//             m_mat(1, m_size * sizeof(T), CV_TYPE, vec.data())
//     {
//     }
//     explicit CvVector(T* vecData, size_t size) :
//             m_size(size),
//             m_mat(1, m_size * sizeof(T), CV_TYPE, vecData)
//     {
//     }

//     [[nodiscard]] cv::UMat &umat()
//     {
//         synchronized = false;
//         m_umat       = m_mat.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
//         return m_umat;
//     }

//     [[nodiscard]] cv::ocl::KernelArg kernelArg()
//     {
//         synchronized = false;
//         m_umat       = m_mat.getUMat(cv::ACCESS_RW, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
//         return cv::ocl::KernelArg::ReadWrite(m_umat);
//     }

//     std::span<T> result()
//     {
//         return {reinterpret_cast<T *>(m_umat.getMat(cv::ACCESS_READ).data), m_size};
//     }

//     void get(cv::Mat& mat)
//     {
//         m_mat = m_umat.getMat(cv::ACCESS_READ);
//         m_mat.copyTo(mat);
//     }
// };

}// namespace ORB_SLAM3::opencl
