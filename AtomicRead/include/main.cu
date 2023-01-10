#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cuda_profiler_api.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "helper.h"

#include "gtest/gtest.h"

int size = 1024;

__global__ void addZeroKernel(int* d_in, int* d_out, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        d_out[tid] = atomicAdd(d_in, 0);
    }
}

__global__ void noCacheKernel(int* d_in, int* d_out, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        asm volatile("ld.global.cg.s32 %0, [%1];"
                     : "=r"(d_out[tid])
                     : "l"(d_in));
    }
}

TEST(Test, addZero)
{
    const int block_size = 256;

    int val = 55;

    thrust::host_vector<int> h_vec(1, val);

    thrust::device_vector<int> d_in(1, val);
    thrust::device_vector<int> d_out(size);

    CUDA_ERROR(cudaProfilerStart());

    addZeroKernel<<<DIVIDE_UP(size, block_size), block_size>>>(
        d_in.data().get(), d_out.data().get(), size);

    auto err = cudaDeviceSynchronize();
    
    CUDA_ERROR(cudaProfilerStop());

    EXPECT_EQ(err, cudaSuccess);

    h_vec = d_out;
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(h_vec[i], val);
    }
}

TEST(Test, noCache)
{
    const int block_size = 256;

    int val = 55;

    thrust::host_vector<int> h_vec(1, val);

    thrust::device_vector<int> d_in(1, val);
    thrust::device_vector<int> d_out(size);

    CUDA_ERROR(cudaProfilerStart());

    noCacheKernel<<<DIVIDE_UP(size, block_size), block_size>>>(
        d_in.data().get(), d_out.data().get(), size);

    auto err = cudaDeviceSynchronize();

    CUDA_ERROR(cudaProfilerStop());

    EXPECT_EQ(err, cudaSuccess);

    h_vec = d_out;
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(h_vec[i], val);
    }
}

int main(int argc, char** argv)
{

    ::testing::InitGoogleTest(&argc, argv);

    if (argc == 2) {
        size = std::atoi(argv[1]);
    }

    std::cout << "\n Size = " << size << "\n";

    return RUN_ALL_TESTS();
}
