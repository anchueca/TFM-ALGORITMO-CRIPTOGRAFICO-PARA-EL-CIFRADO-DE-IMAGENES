#ifndef KERNELS_CUH
#define KERNELS_CUH

// CUDA headers first
#include <cuda_runtime.h>

// Standard headers
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
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
template <typename T> __device__ __forceinline__ T chaotic_function(T x, T r) {
  T t = r + 3.0 * x * x;
  return fabs(cospi(r * cospi(t) * t));
  // return 4.0f * x * (1.0f - x);
}

template <>
__device__ __forceinline__ float chaotic_function<float>(float x, float r) {
  float t = r + 3.0f * x * x;
  return fabsf(cospi(r * cospi(t) * t));
  // return 4.0f * x * (1.0f - x);
}

/**
 * @brief Result of `coupled_map`: mixed value and updated 16-bit CA state.
 */
struct CoupledResult {
  Real mixed;
  unsigned short new_ca;
};

/**
 * @brief Coupled map helper that returns both the mixed value and the
 * evolved automata state by value to avoid taking addresses of local
 * variables (which can force spills to local memory).
 */
__device__ __forceinline__ CoupledResult coupled_map(Real c_next, Real *r_next,
                                                     Real *l_next,
                                                     unsigned short ca_state);

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
 * @param d_r_params Per-seed chaotic parameter array (values in [0,1], scaled
 * to [3,7] in kernel).
 * @param total_steps Total number of evolution steps (transition + rows).
 * @param d_chaotic_values Output buffer for chaotic values used in
 * permutations.
 * @param permutation_block_size Size of the block (squared) for permutation
 * generation logic.
 * @param transition_length Number of initial steps to discard (warmup).
 */
__global__ void keystream_generation_parallel(
    unsigned char *__restrict__ d_flow, Real *__restrict__ d_seeds,
    unsigned short *__restrict__ cellular_automata,
    unsigned short *__restrict__ d_image_automata_state,
    Image_dimensions img_dimensions, const Real *__restrict__ d_r_params,
    const size_t total_steps,
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

__global__ void fused_permutation_xor_kernel(
    const unsigned char *__restrict__ image_in,
    unsigned char *__restrict__ image_out,
    const unsigned char *__restrict__ flow,
    const unsigned int *__restrict__ permutation,
    const unsigned int *__restrict__ permutation_inverse,
    const unsigned int *__restrict__ perm_blocks,
    const unsigned int *__restrict__ perm_blocks_inv, size_t block_size,
    const size_t img_dim, bool use_xor, bool inverse_order);

/**
 * @brief Kernel to generate chaotic values from cellular automata states.
 */
__global__ void generate_automata_chaotic(const unsigned int *d_automata_state,
                                          unsigned short *d_chaotic_values,
                                          unsigned int *indices,
                                          size_t block_length);

/**
 * @brief De-interleaves a 3-channel image (BGRBGR...) into a planar
 * horizontally stacked layout.
 */
__global__ void deinterleave_channels_kernel(const unsigned char *input,
                                             unsigned char *output, int width,
                                             int height);

/**
 * @brief Interleaves a planar horizontally stacked image (B...G...R...) into a
 * 3-channel layout.
 */
__global__ void interleave_channels_kernel(const unsigned char *input,
                                           unsigned char *output, int width,
                                           int height);

__global__ void global_seed_mix_kernel(Real *d_seeds, size_t offset,
                                       size_t n_blocks);

#endif // KERNELS_CUH
