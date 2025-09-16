# ifndef KERNELS_CUH
# define KERNELS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "kernels_aux.cuh"

__device__ double uno(double x, double r);

__global__ void flow_encrypt_recursive(
    unsigned char *image,
    float *seeds,
    int width,
    int height,
    float r,
    int rounds
);

__global__ void permute_blocks_kernel(
    unsigned char *input,
    unsigned char *output,
    int *permutations,
    int block_height,
    int block_width,
    int image_rows,
    int image_cols,
    int num_blocks_row,
    int num_blocks_col,
    int channels
);

__host__ void block_phase_permutation(
    cv::Mat& image, int num_rows, int num_cols, int block_height, int block_width, const std::vector<int>& block_permutations
);

__global__ void generate_chaotic(unsigned char* passwords, int num_blocks, double* chaotic_vals, int* indices, double r, int block_length, int transition_length);

__global__ void invert_permutations_kernel(int* permutations, int* inverses, int length);

__host__ std::vector<std::vector<int>> generate_permutations(const std::vector<unsigned char> block_passwords, int block_length, int num_blocks);

# endif // KERNELS_CUH