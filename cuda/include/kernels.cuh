#ifndef KERNELS_CUH
#define KERNELS_CUH

// CUDA headers first
#include <cstddef>
#include <cuda_runtime.h>

// Standard headers
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <limits.h>
#include <vector>

// Project headers
#include "automata.cuh"
#include "kernels_aux.cuh"
#include "structs.cuh"

/**
 * @brief Logistic-map-like chaotic function used by the flow generator.
 *
 * The exact function is implemented in the corresponding source file. It
 * computes the next chaotic value from x using parameter r.
 *
 * @param x Current state value.
 * @param r Chaotic parameter.
 * @return The next chaotic value.
 */
template <typename T> __device__ __forceinline__ T uno(T x, T r) {
  T t = r + 3.0 * x * x;
  return fabs(cos(3.14159265 * r * cos(3.14159265 * t) * t));
}

template <> __device__ __forceinline__ float uno<float>(float x, float r) {
  float t = r + 3.0f * x * x;
  return fabsf(cosf(3.14159265f * r * cosf(3.14159265f * t) * t));
}

/**
 * @brief Kernel that generate the keystrea
 *
 * This kernel evolves a chaotic map driven by seeds to produce a flow.
 *
 * @param keystream_out output matrix with the generated keystream.
 * @param width Matrix width (columns).
 * @param height Matrix height (rows).
 * @param r Chaotic map parameter.
 */
// 24 bits precission only
template <typename T>
__global__ void keystream_generation(unsigned char *__restrict__ d_flow,
                                     T *__restrict__ d_seeds,
                                     Image_dimensions img_dimensions, T r,
                                     size_t transition_length) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= img_dimensions.cols)
    return;

  T xn = d_seeds[x];

  int idx = x;
  int stride = img_dimensions.cols;

  for (int k = 0; k < transition_length; k++) {
    xn = uno<T>(xn, r);
  }

  for (int y = 0; y < img_dimensions.rows; y++) {
    xn = uno<T>(xn, r);

    d_flow[idx] = convertToBitStream(xn);

    idx += stride;
  }

  d_seeds[x] = xn;
}

/**
 * @brief Kernel to generate keystream using a Coupled Map Lattice (CML) in
 * parallel.
 *
 * This kernel implements the CML system where each cell's next state depends on
 * its current state and the states of its neighbors (coupling). This provides
 * spatial diffusion in the generated keystream.
 *
 * @tparam T The floating-point type for the chaotic map.
 * @param d_flow Output buffer for the generated keystream.
 * @param d_seeds Input/Output buffer for the seeds/state of the map.
 * @param img_dimensions Struct containing the image dimensions.
 * @param r The chaotic parameter.
 * @param position The current row index being processed (used for coupling with
 * the previous row).
 */
template <typename T> // Coupled map lattice
__global__ void keystream_generation_parallel(
    unsigned char *__restrict__ d_flow, T *__restrict__ d_seeds,
    Image_dimensions img_dimensions, T r, size_t position) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= img_dimensions.cols)
    return;

  int current_idx = position * img_dimensions.cols + x;

  T left_xn = (x > 0) ? d_seeds[x - 1] : d_seeds[img_dimensions.cols - 1];
  T previous_xn = d_seeds[x];
  T right_xn = (x < img_dimensions.cols - 1) ? d_seeds[x + 1] : d_seeds[0];

  T coupled_xn = (previous_xn + left_xn + right_xn)/3;

  coupled_xn = uno<T>(coupled_xn, r);

  if (d_flow != nullptr)
    d_flow[current_idx] = convertToBitStream(coupled_xn);

  d_seeds[x] = coupled_xn;
}

/**
 * @brief Kernel to convert raw bits (integers) into normalized floating-point
 * seeds.
 *
 * This kernel takes an array of raw integer data (interpreted from the password
 * bytes) and normalizes each element to the range [0, 1] to be used as initial
 * seeds for the chaotic maps.
 *
 * @tparam T The floating-point type (float or double).
 * @param d_seeds Pointer to the device memory containing the raw integers
 * (in-place conversion).
 * @param num_elements The total number of elements to convert.
 */
template <typename T>
__global__ void convert_bits_to_real_kernel(T *d_seeds, size_t num_elements) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= num_elements)
    return;
  d_seeds[idx] =
      static_cast<T>(reinterpret_cast<uint32_t *>(d_seeds)[idx]) / UINT_MAX;
}

/**
 * @brief Kernel that permutes image blocks according to provided permutations.
 *
 * @param image Input image buffer on device.
 * @param image_out Output image buffer on device.
 * @param permutations Flattened array of block permutations.
 * @param block_size Size of a square block (in pixels per side).
 * @param cols Number of columns of blocks.
 * @param rows Number of rows of blocks.
 */
__global__ void permute_blocks_kernel(unsigned char *image,
                                      unsigned char *image_out,
                                      unsigned int *permutations,
                                      size_t block_size,
                                      Image_dimensions img_dimensions);

/**
 * @brief Performs intra-block pixel permutation using a checkerboard pattern
 * selection.
 *
 * This kernel divides the image into square blocks of size `block_size`. For
 * each pixel, it calculates its target position within the block based on a
 * pre-computed permutation table. To increase cryptographic
 * confusion/diffusion, the specific permutation table used alternates between a
 * forward `permutation` and an `permutation_inverse` based on the block's grid
 * coordinates (a checkerboard/parity pattern).
 *
 * @note This kernel implements a "Gather" approach: threads map to the
 * *destination* (x,y) and calculate where to read the *source* pixel from. This
 * ensures the write operation to global memory is coalesced.
 *
 * @param image             Pointer to the source image data (device memory).
 * @param image_out         Pointer to the destination image data (device
 * memory).
 * @param permutation       Pointer to the primary permutation array (flat array
 * of size block_size^2).
 * @param permutation_inverse Pointer to the secondary/inverse permutation array
 * (flat array of size block_size^2).
 * @param block_size        The width/height of the square blocks (e.g., 16,
 * 32).
 * @param img_dimensions    Struct containing the image dimensions (.rows and
 * .cols).
 */
__global__ void permute_blocks_kernel_simple(unsigned char *image,
                                             unsigned char *image_out,
                                             unsigned int *permutation,
                                             unsigned int *permutation_inverse,
                                             size_t block_size,
                                             Image_dimensions img_dimensions);

/**
 * @brief Kernel to generate chaotic values used for ordering/permutations.
 *
 * Each password segment produces a sequence of chaotic values which are used
 * together with indices to create permutations for blocks.
 *
 * @param passwords Password segments (one per block) on device.
 * @param num_blocks Number of blocks/password segments.
 * @param chaotic_vals Output chaotic values array on device (flattened).
 * @param indices Output indices associated with chaotic values.
 * @param r Chaotic parameter.
 * @param block_length Length of each block/password segment.
 * @param transition_length Number of transition values used for permutation
 * generation.
 */
template <typename T>
__global__ void generate_chaotic(unsigned int *passwords, size_t num_blocks,
                                 T *chaotic_vals, unsigned int *indices, T r,
                                 size_t block_length,
                                 size_t transition_length) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_blocks)
    return;

  T x = (static_cast<T>(passwords[idx]) + 1.0) /
        (UINT_MAX * 1.0); // Normalize to (0,1)

  for (int i = 0; i < transition_length; ++i) {
    x = uno<T>(x, r);
  }

  int base_idx = idx * block_length;
  for (int i = 0; i < block_length; i++) {
    x = uno<T>(x, r);
    chaotic_vals[base_idx + i] = x;
    indices[base_idx + i] = i;
  }

  sort_indices_by_chaotic_values<T>(base_idx, chaotic_vals, indices,
                                    block_length);
}

/**
 * @brief Kernel that permutes columns of the image according to a permutation.
 *
 * @param image Input image buffer on device.
 * @param image_out Output image buffer on device.
 * @param permutation Column permutation array on device.
 * @param cols Number of columns.
 * @param rows Number of rows.
 */
__global__ void permute_columns_kernel(unsigned char *image,
                                       unsigned char *image_out,
                                       unsigned int *permutation,
                                       Image_dimensions img_dimensions);

/**
 * @brief Kernel that permutes rows of the image according to a permutation.
 *
 * @param image Input image buffer on device.
 * @param image_out Output image buffer on device.
 * @param permutation Row permutation array on device.
 * @param cols Number of columns.
 * @param rows Number of rows.
 */
__global__ void permute_rows_kernel(unsigned char *image,
                                    unsigned char *image_out,
                                    unsigned int *permutation,
                                    Image_dimensions img_dimensions);

/**
 * @brief Kernel to generate chaotic values from cellular automata states.
 *
 * The kernel consumes pointers to automata states and reduces them to
 * short chaotic values which are stored in d_chaotic_values. Indices are
 * prepared for subsequent sorting.
 *
 * @param automata_states Array of device pointers to automata packed states.
 * @param d_chaotic_values Output array of reduced chaotic values on device.
 * @param num_blocks Number of automata/blocks.
 * @param indices Output indices array associated with chaotic values.
 * @param block_length Length of each block used for reduction.
 */
__global__ void generate_automata_chaotic(unsigned int **automata_states,
                                          unsigned short *d_chaotic_values,
                                          size_t num_blocks,
                                          unsigned int *indices,
                                          size_t block_length);

/**
 * @brief De-interleaves a 3-channel image (BGRBGR...) into a planar horizontally stacked layout (B...G...R...).
 * Mapping: Input(x,y, c) -> Output(x + c*width, y)
 */
__global__ void deinterleave_channels_kernel(const unsigned char *input,
                                             unsigned char *output,
                                             int width, int height);

/**
 * @brief Interleaves a planar horizontally stacked image (B...G...R...) into a 3-channel layout (BGRBGR...).
 * Mapping: Input(x + c*width, y) -> Output(x,y, c)
 */
__global__ void interleave_channels_kernel(const unsigned char *input,
                                           unsigned char *output,
                                           int width, int height);

#endif // KERNELS_CUH