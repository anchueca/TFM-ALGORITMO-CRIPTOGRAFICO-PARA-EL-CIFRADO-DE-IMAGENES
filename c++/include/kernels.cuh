# ifndef KERNELS_CUH
# define KERNELS_CUH

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <iostream>

#include "kernels_aux.cuh"

__device__ double uno(double x, double r);

/*__global__ void flow_encrypt_recursive(
    unsigned char *image,
    float *seeds,
    int width,
    int height,
    float r,
    int rounds
);*/

__global__ void permute_blocks_kernel(
    unsigned char *image, int *permutations,
    size_t block_size, size_t blocks_per_row);

__host__ void block_phase_permutation(
    cv::cuda::GpuMat image, std::vector<std::vector<int>>& block_permutations);

__global__ void generate_chaotic(
    unsigned char* passwords, size_t num_blocks, double* chaotic_vals,
    int* indices, double r, size_t block_length, size_t transition_length);

/*__global__ void invert_permutations_kernel(
    int* permutations, int* inverses, int length);*/

__host__ std::vector<std::vector<int>> generate_permutations(
    const std::vector<unsigned char> block_passwords, size_t block_length,
    size_t num_blocks);

# endif // KERNELS_CUH