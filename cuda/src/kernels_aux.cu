/**
 * @file kernels_aux.cu
 * @brief Implementations of small CUDA kernels used for image tiling and
 * permutation helpers.
 */

#include "../include/kernels_aux.cuh"

__global__ void
invert_permutations_kernel(const unsigned int *__restrict__ permutations,
                           unsigned int *__restrict__ inverses,
                           size_t block_length) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < block_length; i += stride) {
    unsigned int new_pos = permutations[i];
    inverses[new_pos] = i;
  }
}

__device__ unsigned char convertToBitStream(double value) {
  // Use the least significant 32 bits of the double's representation
  unsigned int x = (unsigned int)__double_as_longlong(value);
  // Binary XOR reduction: 32 -> 16 -> 8 bits
  x ^= (x >> 16);
  x ^= (x >> 8);
  return (unsigned char)(x & 0xFF);
}

__device__ unsigned char convertToBitStream(float value) {
  unsigned int x = __float_as_uint(value);
  // Binary XOR reduction
  x ^= (x >> 16);
  x ^= (x >> 8);
  return (unsigned char)(x & 0xFF);
}