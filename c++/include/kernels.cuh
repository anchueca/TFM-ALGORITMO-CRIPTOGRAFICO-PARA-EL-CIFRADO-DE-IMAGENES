# ifndef KERNELS_CUH
# define KERNELS_CUH

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>

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
    unsigned char *image, unsigned char *image_out, int *permutations,
    size_t block_size, size_t cols, size_t rows);

__global__ void generate_chaotic(
    unsigned char* passwords, size_t num_blocks, double* chaotic_vals,
    int* indices, double r, size_t block_length, size_t transition_length);

__global__ void invert_permutations_kernel(
    int* permutations, int* inverses, int length);

__global__ void permute_columns_kernel(
    unsigned char *image, unsigned char *image_out, int *permutation, size_t cols, size_t rows);

__global__ void permute_rows_kernel(
    unsigned char *image, unsigned char *image_out, int *permutation, size_t cols, size_t rows);

# endif // KERNELS_CUH