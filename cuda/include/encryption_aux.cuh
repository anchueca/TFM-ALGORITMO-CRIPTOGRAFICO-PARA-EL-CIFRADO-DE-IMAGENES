#ifndef ENCRYPTION_AUX_CUH
#define ENCRYPTION_AUX_CUH

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "CudaPermutation.cuh"
#include "automata.cuh"
#include "kernels.cuh"
#include "structs.cuh"

/**
 * @brief Generate block permutations from flow passwords used by the flow
 * stage.
 *
 * The function transforms block_passwords into device-side permutation arrays
 * that will be used to permute pixels inside blocks.
 *
 * @param block_passwords Vector containing concatenated password bytes per
 * block.
 * @param block_length Length of each block/password segment.
 * @param num_blocks Number of blocks (password segments).
 * @param transition_length Number of transition elements used to build the
 * permutation.
 * @return Device pointer to the flattened permutations array (caller must
 * free).
 */
__host__ unsigned int *
generate_flow_permutations(const std::vector<unsigned char> block_passwords,
                           size_t block_length, size_t num_blocks,
                           const size_t transition_length);

/**
 * @brief Apply a block-phase permutation to the image on the device.
 *
 * @param d_image Input device image buffer.
 * @param d_image_out Output device image buffer.
 * @param block_permutations Device pointer to block permutations.
 * @param cols Number of block columns.
 * @param rows Number of block rows.
 * @param block_size Size of each block in pixels (side length).
 */
__host__ void block_phase_permutation(unsigned char *d_image,
                                      unsigned char *d_image_out,
                                      unsigned int *block_permutations,
                                      Image_dimnesions img_dimensions,
                                      size_t block_size);

__host__ void block_phase_permutation_simple(unsigned char *d_image,
                                             unsigned char *d_image_out,
                                             unsigned int *permutation,
                                             unsigned int *permutation_inverse,
                                             Image_dimnesions img_dimensions,
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
                                           Image_dimnesions img_dimensions,
                                           bool inverse);

/**
 * @brief Applies the flow encryption stage using provided seeds and chaotic
 * parameter.
 *
 * @param image Device pointer to the input image.
 * @param image_out Device pointer to the output image.
 * @param seeds Flow seeds per block.
 * @param cols Image width in blocks or pixels depending on the pipeline.
 * @param rows Image height in blocks or pixels depending on the pipeline.
 * @param r Chaotic map parameter.
 * @param rounds Number of flow rounds to perform.
 */
__host__ void flow_encrypt(D_pointers &d_pointers,
                           Image_dimnesions img_dimensions);

/**
 * @brief Generate the flow stream stage using provided seeds and chaotic
 * parameter.
 *
 * @param d_flow Device pointer to the output flow.
 * @param d_flow Device pointer to the seeds.
 * @param seeds Flow seeds per block.
 * @param cols Image width in blocks or pixels depending on the pipeline.
 * @param rows Image height in blocks or pixels depending on the pipeline.
 * @param r Chaotic map parameter.
 */
__host__ void generate_flow_stream(D_pointers &d_pointers,
                                   Image_dimnesions img_dimensions, double r);

/**
 * @brief Generate permutations from cellular automata instances.
 *
 * For each provided automaton, this function extracts a packed state and
 * computes a permutation suitable for block reordering.
 *
 * @param automatas Vector of pointers to ElementalCelularAutomata instances.
 * @param steps Number of automata evolution steps used to derive permutations.
 * @param block_length Length of each permutation block.
 * @return Device pointer to the flattened permutations array (caller must
 * free).
 */
__host__ unsigned int *generate_automata_permutations(
    const std::vector<ElementalCelularAutomata *> automatas, const size_t steps,
    const size_t block_length,bool verbose);

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
 * @brief Creates a set of ElementalCelularAutomata instances from password
 * segments.
 *
 * Each password segment initializes the automaton state. Precision level
 * determines how much of the password is used or how states are interpreted.
 *
 * @param password_segments Vector of password byte segments (one per
 * automaton).
 * @param num_blocks Number of automata to create.
 * @param block_size Block size related to the automata cell count.
 * @param precision_level Precision level used when initializing automata
 * states.
 * @return A vector of pointers to created ElementalCelularAutomata instances.
 */
__host__ const std::vector<ElementalCelularAutomata *> createElementalAutomata(
    const std::vector<std::vector<unsigned char>> &password_segments,
    size_t num_blocks, size_t block_size, size_t precision_level);

#endif // ENCRYPTION_AUX_CUH