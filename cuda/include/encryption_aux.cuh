#ifndef ENCRYPTION_AUX_CUH
#define ENCRYPTION_AUX_CUH

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <vector>

#include "CudaPermutation.cuh"
#include "automata.cuh"
#include "kernels.cuh"
#include "structs.cuh"

/**
 * @brief Applies a simplified block permutation to the image.
 *
 * This function applies a permutation to each block of the image. It uses a
 * checkerboard pattern where some blocks use the forward permutation and others
 * use the inverse permutation to increase diffusion.
 *
 * @param d_image Input device image buffer.
 * @param d_image_out Output device image buffer.
 * @param permutation Device pointer to the forward permutation array.
 * @param permutation_inverse Device pointer to the inverse permutation array.
 * @param img_dimensions Struct containing the image dimensions.
 * @param block_size The size of the blocks used for permutation.
 */
__host__ void block_phase_permutation(unsigned char *d_image,
                                      unsigned char *d_image_out,
                                      unsigned int *permutation,
                                      unsigned int *permutation_inverse,
                                      Image_dimensions img_dimensions,
                                      size_t block_size);

/**
 * @brief Executes row and column permutations on the GPU.
 * * @note MEMORY FLOW WARNING:
 * This function performs a "ping-pong" operation.
 * 1. Row Permutation: Input -> Output (buffer)
 * 2. Col Permutation: Output (buffer) -> Input
 * * RESULT: The final permutated image resides in 'd_image' (the input
 * pointer), NOT in 'd_image_out'. 'd_image_out' is used only as a temporary
 * scratchpad.
 * * @param d_image Input image data (and final destination).
 * @param d_image_out Temporary buffer for intermediate step.
 * @param d_row_permutations Device pointer to row permutation vector.
 * @param d_col_permutations Device pointer to col permutation vector.
 * @param img_dimensions Struct containing width and height.
 * @param inverse If true, applies inverse permutations in reverse order (Cols
 * then Rows).
 */
__host__ void rows_and_columns_permutation(unsigned char *d_image,
                                           unsigned char *d_image_out,
                                           unsigned int *d_row_permutations,
                                           unsigned int *d_col_permutations,
                                           Image_dimensions img_dimensions,
                                           bool inverse);

/**
 * @brief Inverts a batch of permutations stored on the GPU.
 *
 * Each permutation is a contiguous segment of length block_length in the
 * d_permutations device buffer. This function produces the inverse
 * permutations in-place or in a separate buffer as required by the caller.
 *
 * @param d_permutations Pointer to device pointer(s) representing permutations
 * to invert.
 * @param block_length Length of each permutation block.
 * @param num_blocks Number of permutations (blocks) to invert.
 */
__host__ void inverse_permutations(unsigned int *d_permutations,
                                   unsigned int **d_permutations_inverse,
                                   size_t block_length, size_t num_blocks);

/**
 * @brief Converts password bits into real-valued seeds for chaotic maps.
 *
 * This function takes a segment of the password (bytes) and converts it into
 * floating-point seeds (normalized to [0, 1]) that are used to initialize
 * the chaotic maps.
 *
 * @param password_segment A vector of bytes representing the password segment.
 * @param d_seeds Pointer to the device memory where the generated seeds will be
 * stored. The function allocates this memory.
 */
__host__ void
convert_bits_to_real(const std::vector<unsigned char> &password_segment,
                     Real **d_seeds);

/**
 * @brief Applies the flow encryption stage using provided seeds and chaotic
 * parameter.
 * @param image Device pointer to the input image.
 * @param image_out Device pointer to the output image.
 * @param seeds Flow seeds per block.
 * @param cols Image width in blocks or pixels depending on the pipeline.
 * @param rows Image height in blocks or pixels depending on the pipeline.
 * @param r Chaotic map parameter.
 * @param rounds Number of flow rounds to perform.
 */
__host__ void flow_encrypt(D_pointers &d_pointers,
                           Image_dimensions img_dimensions);

/**
 * @brief Generates the flow keystream in parallel using a Block-Parallel CML.
 *
 * This function oversees the generation of the chaotic keystream. It configures
 * the kernel launch to process the entire image height in a single pass (using
 * looping inside the kernel) and manages the shared memory allocation required
 * for block-level coupling.
 *
 * @param d_pointers Struct containing device pointers (d_flow, d_seeds).
 * @param img_dimensions Struct containing the image dimensions.
 * @param params Encryption configuration parameters.
 */
__host__ void generate_flow_stream_parallel(D_pointers &d_pointers,
                                            Image_dimensions img_dimensions,
                                            EncryptionParams params);

/**
 * @brief Generate permutation for blocks.
 *
 * This function generates a permutation for blocks based on the provided
 * parameters.
 *
 * @param d_pointers Struct containing device pointers for image data and
 * permutations.
 * @param img_dimensions Struct containing the image dimensions.
 * @param params Struct containing configuration for encryption (block size,
 * rounds, etc.).
 */
__host__ void generate_permutation_block(D_pointers &d_pointers,
                                         Image_dimensions img_dimensions,
                                         EncryptionParams params);

/**
 * @brief Generate permutation from a cellular automaton.
 *
 * This function evolves the automaton for the specified number of steps,
 * extracts its state, and computes a permutation suitable for reordering.
 *
 * @param automata Pointer to ElementalCelularAutomata instance.
 * @param steps Number of automata evolution steps used to derive permutation.
 * @param block_length Length of the permutation.
 * @param verbose Enable verbose output for timing information.
 * @return Device pointer to the permutation array (caller must free).
 */
__host__ unsigned int *
generate_automata_permutations(ElementalCelularAutomata *automata,
                               const size_t steps, const size_t block_length,
                               bool verbose);

/**
 * @brief Unstacks an interleaved (BGR) image on the device into a planar
 * format. Wrapper for deinterleave_channels_kernel.
 */
__host__ void unstack_channels_gpu(unsigned char *d_interleaved,
                                   unsigned char *d_planar, int width,
                                   int height);

/**
 * @brief Stacks a planar image on the device into an interleaved (BGR) format.
 * Wrapper for interleave_channels_kernel.
 */
__host__ void stack_channels_gpu(unsigned char *d_planar,
                                 unsigned char *d_interleaved, int width,
                                 int height);

#endif // ENCRYPTION_AUX_CUH