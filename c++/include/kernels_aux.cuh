#ifndef KERNELS_AUX_CUH
#define KERNELS_AUX_CUH

// CUDA headers first
#include <cuda_runtime.h>

// Standard headers
#include <type_traits>

#include "structs.cuh"

#define MAX_BLOCK_SIZE 64

template <typename T>
__device__ void sort_indices_by_chaotic_values(int base_idx, T *chaotic_vals,
                                               unsigned int *indices,
                                               size_t block_length) {
  
  T local_vals[MAX_BLOCK_SIZE];
  unsigned int local_indices[MAX_BLOCK_SIZE];

  if (block_length > MAX_BLOCK_SIZE) return; 

  for (size_t i = 0; i < block_length; i++) {
      local_vals[i] = chaotic_vals[base_idx + i];
      local_indices[i] = indices[base_idx + i];
  }

  for (size_t i = 1; i < block_length; i++) {
    T key_val = local_vals[i];
    unsigned int key_idx = local_indices[i];

    int j = (int)i - 1;

    while (j >= 0 && local_vals[j] > key_val) {
      local_vals[j + 1] = local_vals[j];
      local_indices[j + 1] = local_indices[j];
      j = j - 1;
    }

    local_vals[j + 1] = key_val;
    local_indices[j + 1] = key_idx;
  }

  for (size_t i = 0; i < block_length; i++) {
      indices[base_idx + i] = local_indices[i];
      chaotic_vals[base_idx + i] = local_vals[i]; 
  }
}

/**
 * @brief Kernel that sorts index arrays per block using associated chaotic
 * values.
 *
 * Each thread (or logical index determined by the grid) handles a separate
 * block segment of length block_length. It calls the device insertion sort to
 * reorder the chaotic values and indices in-place.
 *
 * @tparam T Numeric type of chaotic values.
 * @param d_chaotic_values Device pointer to chaotic values (contiguous blocks).
 * @param num_blocks Number of blocks (segments) to sort.
 * @param indices Device pointer to flat indices array to reorder.
 * @param block_length Length of each block segment.
 */
template <typename T>
__global__ void sort_indices_by_chaotic_values_global(T *d_chaotic_values,
                                                      size_t num_blocks,
                                                      unsigned int *indices,
                                                      size_t block_length) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= (int)num_blocks)
    return;
  int base_idx = idx * (int)block_length;

  sort_indices_by_chaotic_values<T>(base_idx, d_chaotic_values, indices,
                                    block_length);
}

/**
 * @brief Kernel that merges and stacks image tiles.
 *
 * Implementation merges multiple source tiles into a stacked destination image
 * layout. Parameters are kept generic to match the project usage.
 *
 * @param src Source image tiles.
 * @param dst Destination stacked image buffer.
 * @param dst_width Width of the destination image.
 * @param dst_height Height of the destination image.
 */
__global__ void merge_and_stack_kernel(const unsigned char *src,
                                       unsigned char *dst, int dst_width,
                                       int dst_height);

/**
 * @brief Kernel that splits and concatenates an image into tiles.
 *
 * This kernel performs the inverse operation of merge_and_stack_kernel,
 * splitting a source image into tiles and concatenating them into the
 * destination buffer arranged in a specific order.
 *
 * @param src Source image buffer.
 * @param dst Destination tiled/concatenated image buffer.
 * @param width Width of the source image.
 * @param height Height of the source image.
 */
__global__ void split_and_concat_kernel(const unsigned char *src,
                                        unsigned char *dst, int width,
                                        int height);
/**
 * @brief Kernel to invert a batch of permutations in parallel.
 * Each CUDA block is responsible for inverting one permutation.
 * @param permutations Input array of permutations on the GPU.
 * @param inverses Output array for the inverted permutations on the GPU.
 * @param block_length The length of a single permutation.
 * @param num_blocks (Unused) The total number of permutations. The kernel
 * deduces this from the grid size.
 */
/**
 * @brief Kernel to invert a batch of permutations in parallel.
 *
 * Each CUDA block is responsible for inverting one permutation.
 *
 * @param d_permutations Input array of permutations on the GPU (flattened).
 * @param inverses Output array for the inverted permutations on the GPU
 * (flattened).
 * @param block_length Length of a single permutation.
 * @param num_blocks Number of permutations in the batch.
 */
__global__ void invert_permutations_kernel(unsigned int *d_permutations,
                                           unsigned int *inverses,
                                           size_t block_length,
                                           size_t num_blocks);

/**
 * @brief XOR an image buffer with a keystream buffer in-place (per-pixel XOR).
 *
 * This CUDA kernel applies a simple XOR operation between a keystream and an
 * image buffer. The kernel is intended to be launched with a 2D grid matching
 * the image dimensions (or a flattened 1D grid treating width*height as
 * length).
 *
 * @param keystream Device pointer to keystream bytes (one byte per pixel).
 * @param image Device pointer to image bytes (will be modified in-place).
 * @param width Image width in pixels (columns).
 * @param height Image height in pixels (rows).
 */
__global__ void image_xor(unsigned char *keystream, unsigned char *image,
                          Image_dimnesions img_dimensions);

#endif // KERNELS_AUX_CUH
