#ifndef CUDA_PERMUTATION_CUH
#define CUDA_PERMUTATION_CUH
#include <cstdint>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "structs.cuh"

// =================================================================================
//                            SINGLE ARRAY BITONIC SORT
// =================================================================================

/**
 * @brief Generates a permutation (argsort) based on a chaotic sequence using
 * the GPU.
 * * Performs a parallel Bitonic Sort. Automatically handles padding if n is not
 * a power of 2.
 * * @param h_chaotic_sequence Pointer to the Real array containing chaotic
 * values (Host/CPU).
 * @param h_permutation Pointer to the int array where indices will be stored
 * (Host/CPU).
 * @param n Number of elements.
 */
void compute_permutation_gpu(const Real *h_chaotic_sequence,
                             int *h_permutation, int n);

/**
 * @brief Generates a permutation (argsort) based on a chaotic sequence already
 * on the GPU. Accepts optional pooled scratch buffers for zero-allocation runtime.
 */
void compute_permutation_device(Real *d_values, unsigned int *d_indices,
                                int n, Real *d_padded_values_pool = nullptr,
                                unsigned int *d_padded_indices_pool = nullptr);

/**
 * @brief GPU Kernel: Single-pass Shared Memory Bitonic Sort for small arrays (n <= 1024).
 */
__global__ void bitonic_sort_shared_kernel(const Real *__restrict__ d_values,
                                           unsigned int *__restrict__ d_indices,
                                           int n, int padded_size);

/**
 * @brief GPU Kernel: Initializes buffers.
 * Copies indices 0..n-1 and fills the padding area with a large value.
 */
__global__ void init_buffers_kernel(Real *values, unsigned int *indices, int n,
                                    int padded_size);

/**
 * @brief GPU Kernel: Executes one step of the Bitonic Sort.
 */
__global__ void bitonic_sort_step_kernel(Real *values, unsigned int *indices, int j,
                                         int k, int padded_size);

// =================================================================================
//                            BATCHED BITONIC SORT
// =================================================================================

/**
 * @brief Host function that orchestrates the Batched Bitonic Sort.
 * * Sorts multiple independent blocks of data simultaneously.
 * * Accepts optional pooled scratch buffers for zero-allocation runtime.
 */
void batched_gpu_argsort(unsigned short *d_keys, unsigned int *d_indices,
                         size_t num_blocks, size_t block_len,
                         int *d_padded_keys_pool = nullptr,
                         unsigned int *d_padded_indices_pool = nullptr);

/**
 * @brief GPU Kernel: Copies real data to a padded buffer (power of 2).
 * Converts unsigned short to int to allow using INT_MAX as the padding
 * sentinel.
 */
__global__ void copy_to_padded_kernel(const unsigned short *input_vals,
                                      const unsigned int *input_idxs,
                                      int *padded_vals,
                                      unsigned int *padded_idxs, int valid_len,
                                      int padded_len, int total_blocks);

/**
 * @brief GPU Kernel: One step of Bitonic Sort applied to multiple blocks.
 */
__global__ void batched_bitonic_step_kernel(int *values, unsigned int *indices,
                                            int j, int k, int padded_len,
                                            int total_blocks);

/**
 * @brief GPU Kernel: Copies valid results back from the padded buffer to the
 * original buffer.
 */
__global__ void copy_from_padded_kernel(const int *padded_vals,
                                        const unsigned int *padded_idxs,
                                        unsigned int *output_idxs,
                                        int valid_len, int padded_len,
                                        int total_blocks);

// =================================================================================
//                                  HELPERS
// =================================================================================

/**
 * @brief Calculates the next power of 2 for a given number.
 */
int next_power_of_2(int n);

#endif // CUDA_PERMUTATION_CUH