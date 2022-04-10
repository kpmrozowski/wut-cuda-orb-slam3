#ifdef __CUDACC__
#define DEVHOST __device__ __host__
#else
#define DEVHOST
#endif

#include <cudatest/cudatest.h>
#include <stdio.h>
#include <iostream>
#include <utility>
#include <array>
#include <list>
#include <unordered_map>
#include <thrust/device_vector.h>

namespace cudatest {

class Point {
public:
    int x, y;
    DEVHOST Point();
    DEVHOST Point(int a,int b) = delete;
    DEVHOST Point(const Point &&);
    DEVHOST Point(const Point &) = delete;
    DEVHOST Point(Point &) = delete;
    std::array<int, 2> arr{};
};
DEVHOST Point::Point() { std::printf("point\n"); };
// Point::Point(int a,int b) : x(std::move(a)), y(std::move(b)) { printf("param\n"); arr[0] = a; arr[1] = b; }
DEVHOST Point::Point(const Point &&p) : x(p.x), y(p.y), arr(p.arr) { printf("moved\n"); }

__global__ void init_point(void* buffer, int a, int b) {
    Point p;
    p.x = a;
    p.y = threadIdx.x;
    p.arr = std::array<int, 2>{p.x, p.y};
    // p.m[threadIdx.x] = a;
    // p.m[threadIdx.x] = b;
    if (threadIdx.x == 2){
        new(buffer) Point{std::move(p)};
        for (int i = 0; i < p.arr.size(); ++i)
            printf("p.arr[0]=%d\n", p.arr[i]);
    }
}

__global__ void add(float* v1, float* v2, int N) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // one thread 
    if (thread_id < N) {
        v1[thread_id] = v1[thread_id] + v2[thread_id];
    }
}

void test_cuda() {
    int count = 0;
    int i = 0;
    
    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return;
    }
    int cuda_count = 0;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) { cuda_count++;}
            std::cout << "[" << i << "] --" << prop.name << std::endl;
        }
    }
    if(cuda_count == 0) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return;
    }
    void* buff;
    cudaMalloc(&buff,sizeof(Point));
    init_point<<<1,65>>>(buff,11,20);
    cudaThreadSynchronize();
    Point cpu_point;
    cudaMemcpy(&cpu_point,buff,sizeof(Point),cudaMemcpyDeviceToHost);
    std::cout << cpu_point.x << std::endl;
    std::cout << cpu_point.y << std::endl;
   
    const int N = (1 << 20);
   // std::array<float, N> vec1(1.0f), vec2(2.0f);
   float* h_vec1 = new float[N];
   float* h_vec2 = new float[N];
   float* d_vec1;
   float* d_vec2;
   cudaMalloc(&d_vec1, N * sizeof(float));
   cudaMalloc(&d_vec2, N * sizeof(float));
   for (int i = 0; i < N; ++i) {
       h_vec1[i] = 1.0f;
       h_vec2[i] = 2.0f;
   }

   cudaMemcpy(d_vec1, h_vec1, N * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_vec2, h_vec2, N * sizeof(float), cudaMemcpyHostToDevice);

   // cudaMallocMenaged(&h_vec1, N * sizeof(float));
   // cudaMallocMenaged(&h_vec1, N * sizeof(float));

   dim3 num_threads(1024);
   dim3 num_blocks(N / 1024 + 1);

   add<<<num_blocks, num_threads>>>(d_vec1, d_vec2, N);
   cudaDeviceSynchronize();
   cudaMemcpy(h_vec1, d_vec1, N * sizeof(float), cudaMemcpyDeviceToHost);

   for (int i = 0; i < N; ++i) {
       if (h_vec1[i] != 3) {
           std::cout << "Incorrect result!" << std::endl;
           return;
       }
   }
   std::cout << "Correct result!" << std::endl;
   delete[] h_vec1;
   delete[] h_vec2;
   cudaFree(d_vec1);
   cudaFree(d_vec2);
}

} // namespace cudatest

int main() {
   cudatest::test_cuda();
}
