#ifndef KERNELS_AUX_CUH
#define KERNELS_AUX_CUH

// CUDA headers first
#include <cuda_runtime.h>

// Standard headers
#include <stdio.h>
#include <type_traits>

#include "structs.cuh"

#define MAX_BLOCK_SIZE 64

/**
 * @brief Sorts indices within a block based on chaotic values using insertion
 * sort.
 *
 * This device function performs an in-place insertion sort on a local array of
 * chaotic values and their corresponding indices. It is used to generate
 * permutations for a specific block.
 *
 * @param base_idx The global base index for the current block in the input
 * arrays.
 * @param chaotic_vals Pointer to the global array of chaotic values.
 * @param indices Pointer to the global array of indices to be sorted.
 * @param block_length The number of elements in the block to be sorted.
 */
__device__ void sort_indices_by_chaotic_values(int base_idx, Real *chaotic_vals,
                                               unsigned int *indices,
                                               size_t block_length);
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
 * @brief Kernel that sorts index arrays per block using associated chaotic
 * values.
 *
 * Each thread (or logical index determined by the grid) handles a separate
 * block segment of length block_length. It calls the device insertion sort to
 * reorder the chaotic values and indices in-place.
 *
 * @param d_chaotic_values Device pointer to chaotic values (contiguous blocks).
 * @param num_blocks Number of blocks (segments) to sort.
 * @param indices Device pointer to flat indices array to reorder.
 * @param block_length Length of each block segment.
 */
__global__ void sort_indices_by_chaotic_values_global(Real *d_chaotic_values,
                                                      size_t num_blocks,
                                                      unsigned int *indices,
                                                      size_t block_length);

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
                          Image_dimensions img_dimensions);

#endif // KERNELS_AUX_CUH
