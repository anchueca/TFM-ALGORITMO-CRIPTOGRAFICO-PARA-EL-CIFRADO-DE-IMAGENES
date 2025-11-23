/**
 * @file kernels_aux.cu
 * @brief Implementations of small CUDA kernels used for image tiling and
 * permutation helpers.
 */

#include "../include/kernels_aux.cuh"


__global__ void invert_permutations_kernel(unsigned int *permutations,
                                           unsigned int *inverses,
                                           size_t block_length,
                                           size_t num_blocks) {
  // Each block handles a single permutation inversion. Grid-stride loop covers
  // all elements.
  int permutation_id = blockIdx.x;
  int thread_id_in_block = threadIdx.x;
  int threads_per_block = blockDim.x;

  for (int i = thread_id_in_block; i < block_length; i += threads_per_block) {
    size_t idx_in = permutation_id * block_length + i;
    unsigned int new_pos = permutations[idx_in];
    size_t idx_out = permutation_id * block_length + new_pos;
    inverses[idx_out] = i;
  }
}

__global__ void image_xor(unsigned char *keystream, unsigned char *image,
                          Image_dimnesions img_dimensions) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= img_dimensions.cols || y >= img_dimensions.rows) {
      return;
  }

  int idx = y * img_dimensions.cols + x;

  image[idx] ^= keystream[idx];
}