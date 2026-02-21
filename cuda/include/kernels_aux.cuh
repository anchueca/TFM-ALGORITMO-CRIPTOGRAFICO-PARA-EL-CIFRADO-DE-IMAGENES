#ifndef KERNELS_AUX_CUH
#define KERNELS_AUX_CUH

// CUDA headers first
#include <cuda_runtime.h>

// Standard headers
#include <stdio.h>
#include <type_traits>

#include "structs.cuh"

#define MAX_BLOCK_SIZE 256

#include <iostream>
#include <stdexcept>
#include <string>

/**
 * @brief Utility function to check CUDA return status and throw detailed
 * exception on error.
 *
 * @param err The CUDA error code to check.
 * @param msg The context message to include in the exception.
 */
inline void checkCudaError(cudaError_t err, const std::string &msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(msg + ": " + cudaGetErrorString(err));
  }
}

/**
 * @brief Converts a double precision chaotic value to a single byte keystream
 * value.
 *
 * @param value The input chaotic value (double).
 * @return The corresponding byte value (unsigned char).
 */
__device__ unsigned char convertToBitStream(double value);

/**
 * @brief Converts a single precision chaotic value to a single byte keystream
 * value.
 *
 * @param value The input chaotic value (float).
 * @return The corresponding byte value (unsigned char).
 */
__device__ unsigned char convertToBitStream(float value);

/**
 * @brief Kernel to invert a batch of permutations in parallel.
 *
 * Each CUDA block is responsible for inverting one permutation.
 *
 * @param d_permutations Input array of permutations on the GPU (flattened).
 * @param inverses Output array for the inverted permutations on the GPU
 * (flattened).
 * @param block_length Length of a single permutation.
 */
__global__ void
invert_permutations_kernel(const unsigned int *__restrict__ d_permutations,
                           unsigned int *__restrict__ inverses,
                           size_t block_length);

#endif // KERNELS_AUX_CUH
