#ifndef KERNELS_CUH
#define KERNELS_CUH

// CUDA headers first
#include <cuda_runtime.h>

// Standard headers
#include <iostream>
#include <vector>

// Project headers
#include "automata.cuh"
#include "kernels_aux.cuh"

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
__device__ double uno(double x, double r);

/**
 * @brief Kernel that applies the recursive flow encryption on image blocks.
 *
 * This kernel evolves a chaotic map driven by seeds to produce a flow that is
 * applied to the image pixels across several rounds.
 *
 * @param image Input image buffer in device memory.
 * @param image_out Output image buffer in device memory.
 * @param seeds Per-block seeds used to initialize chaotic maps.
 * @param width Image width (columns).
 * @param height Image height (rows).
 * @param r Chaotic map parameter.
 * @param rounds Number of rounds to apply.
 */
__global__ void keystream_to_image(unsigned char *image,
                                   unsigned char *image_out,
                                   const unsigned char *seeds, int width,
                                   int height, double r, int rounds);

/**
 * @brief Kernel that generate the keystrea
 *
 * This kernel evolves a chaotic map driven by seeds to produce a flow.
 *
 * @param keystream_out output matrix with the generated keystream.
 * @param seeds Per-block seeds used to initialize chaotic maps.
 * @param width Matrix width (columns).
 * @param height Matrix height (rows).
 * @param r Chaotic map parameter.
 * @param rounds Number of rounds to apply.
 */
__global__ void keystream_generation(unsigned char *keystream_out,
                                     const unsigned char *seeds, int width,
                                     int height, double r, int rounds);

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
                                      size_t block_size, size_t cols,
                                      size_t rows);

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
__global__ void generate_chaotic(unsigned char *passwords, size_t num_blocks,
                                 double *chaotic_vals, unsigned int *indices,
                                 double r, size_t block_length,
                                 size_t transition_length);

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
                                       unsigned int *permutation, size_t cols,
                                       size_t rows);

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
                                    unsigned int *permutation, size_t cols,
                                    size_t rows);

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

#endif // KERNELS_CUH