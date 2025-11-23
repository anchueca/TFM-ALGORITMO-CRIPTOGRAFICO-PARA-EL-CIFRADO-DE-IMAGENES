/**
 * @file kernels_aux.cu
 * @brief Implementations of small CUDA kernels used for image tiling and
 * permutation helpers.
 */

#include "../include/kernels_aux.cuh"

// Small kernels for image tiling and permutation helpers.
__global__ void split_and_concat_kernel(const unsigned char *src,
                                        unsigned char *dst, int width,
                                        int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int src_idx = (y * width + x) * 3;

    int dst_width = width * 3;

    int dst_idx_b = y * dst_width + x;
    int dst_idx_g = y * dst_width + x + width;
    int dst_idx_r = y * dst_width + x + 2 * width;

    dst[dst_idx_b] = src[src_idx];
    dst[dst_idx_g] = src[src_idx + 1];
    dst[dst_idx_r] = src[src_idx + 2];
  }
}

__global__ void merge_and_stack_kernel(const unsigned char *src,
                                       unsigned char *dst, int dst_width,
                                       int dst_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dst_width && y < dst_height) {
    int src_width = dst_width * 3;

    int src_idx_b = y * src_width + x;
    int src_idx_g = y * src_width + x + dst_width;
    int src_idx_r = y * src_width + x + 2 * dst_width;

    int dst_idx = (y * dst_width + x) * 3;

    dst[dst_idx] = src[src_idx_b];     // Write B channel
    dst[dst_idx + 1] = src[src_idx_g]; // Write G channel
    dst[dst_idx + 2] = src[src_idx_r]; // Write R channel
  }
}

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