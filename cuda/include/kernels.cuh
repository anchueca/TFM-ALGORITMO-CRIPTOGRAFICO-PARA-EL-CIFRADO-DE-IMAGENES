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
#include "automataKernel.cuh"
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
template <typename T> __device__ __forceinline__ T chaotic_functio(T x, T r) {
  T t = r + 3.0 * x * x;
  return fabs(cos(3.14159265 * r * cos(3.14159265 * t) * t));
}

template <>
__device__ __forceinline__ float chaotic_functio<float>(float x, float r) {

  float t = r + 3.0f * x * x;
  return fabsf(cosf(3.14159265f * r * cosf(3.14159265f * t) * t));
}

__device__ __forceinline__ Real coupled_map(Real c_seed, Real r_seed,
                                            Real l_seed, Real r,
                                            unsigned short *celular_automata);

/**
 * @brief Kernel to generate keystream using a Block-Parallel Coupled Map
 * Lattice (CML).
 *
 * This kernel implements a parallel CML system where each thread block operates
 * independently. The coupling is cyclic within the valid threads of the block.
 * Uses shared memory to store the block's state ("seeds") to minimize global
 * memory access.
 *
 * @param d_flow Output buffer for the generated keystream.
 * @param d_seeds Input/Output buffer for the seeds/state of the map.
 * @param celular_automata Checksum/automata state (unused in this kernel logic
 * but passed).
 * @param img_dimensions Struct containing the image dimensions.
 * @param r The chaotic parameter.
 * @param total_steps Total number of evolution steps (transition + rows).
 * @param d_chaotic_values Output buffer for chaotic values used in
 * permutations.
 * @param permutation_block_size Size of the block (squared) for permutation
 * generation logic.
 * @param transition_length Number of initial steps to discard (warmup).
 */
__global__ void keystream_generation_parallel(
    unsigned char *__restrict__ d_flow, Real *__restrict__ d_seeds,
    unsigned short *celular_automata, unsigned short *d_image_automata_state,
    Image_dimensions img_dimensions, Real r, const size_t total_steps,
    Real *__restrict__ d_chaotic_values_for_permutation,
    size_t permutation_block_size, size_t transition_length, size_t numBlocks);

/**
 * @brief Kernel to convert raw bits (integers) into normalized floating-point
 * seeds.
 *
 * This kernel takes an array of raw integer data (interpreted from the password
 * bytes) and normalizes each element to the range [0, 1] to be used as initial
 * seeds for the chaotic maps.
 *
 * @param d_seeds Pointer to the device memory containing the raw integers
 * (in-place conversion).
 * @param num_elements The total number of elements to convert.
 */
__global__ void convert_bits_to_real_kernel(Real *d_seeds, size_t num_elements);

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
 * @brief De-interleaves a 3-channel image (BGRBGR...) into a planar
 * horizontally stacked layout (B...G...R...). Mapping: Input(x,y, c) ->
 * Output(x + c*width, y)
 */
__global__ void deinterleave_channels_kernel(const unsigned char *input,
                                             unsigned char *output, int width,
                                             int height);

/**
 * @brief Interleaves a planar horizontally stacked image (B...G...R...) into a
 * 3-channel layout (BGRBGR...). Mapping: Input(x + c*width, y) -> Output(x,y,
 * c)
 */
__global__ void interleave_channels_kernel(const unsigned char *input,
                                           unsigned char *output, int width,
                                           int height);

/**
 * @brief Kernel to sort indices based on chaotic values for permutations.
 *
 * Typically called with <<<num_permutations, 1>>>.
 * N is small (e.g. 64), so a simple serial sort per block is sufficient and
 * robust.
 */
__global__ void sort_indices_by_chaotic_values_global(Real *chaotic_values,
                                                      size_t num_permutations,
                                                      unsigned int *indices,
                                                      size_t block_area);

__global__ void global_seed_mix_kernel(Real *d_seeds, size_t offset,
                                       size_t n_blocks);

#endif // KERNELS_CUH
