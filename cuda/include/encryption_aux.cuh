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
template <typename T>
__host__ unsigned int *
generate_flow_permutations(const std::vector<unsigned char> block_passwords,
                           const size_t block_length, const size_t num_blocks,
                           const size_t transition_length, const T r) {
  if (block_passwords.size() < num_blocks) {
    throw std::runtime_error("Insufficient passwords for blocks");
  }

  size_t total_size = num_blocks * block_length;

  unsigned int *d_passwords = nullptr;
  unsigned int *d_indices = nullptr;
  T *d_chaotic_values = nullptr;

  cudaError_t err = cudaMalloc(&d_passwords, num_blocks * sizeof(unsigned int));
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Flow: Failed to allocate device memory for passwords");
  }

  err = cudaMalloc(&d_indices, total_size * sizeof(unsigned int));
  if (err != cudaSuccess) {
    cudaFree(d_passwords);
    throw std::runtime_error("Failed to allocate device memory for indices");
  }

  err = cudaMalloc(&d_chaotic_values, total_size * sizeof(T));
  if (err != cudaSuccess) {
    cudaFree(d_passwords);
    cudaFree(d_indices);
    throw std::runtime_error(
        "Failed to allocate device memory for chaotic values");
  }

  // Copy
  err = cudaMemcpy(d_passwords, block_passwords.data(),
                   num_blocks * sizeof(unsigned int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_passwords);
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);
    throw std::runtime_error("Failed to copy passwords to device");
  }

  const int threadsPerBlock = 256;
  const int numBlocks = (num_blocks + threadsPerBlock - 1) / threadsPerBlock;

  generate_chaotic<T><<<numBlocks, threadsPerBlock>>>(
      d_passwords, num_blocks, d_chaotic_values, d_indices, r, block_length,
      transition_length);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_passwords);
    cudaFree(d_chaotic_values);
    throw std::runtime_error("Kernel execution failed");
  }

  err = cudaDeviceSynchronize();

  if (err != cudaSuccess) {
    cudaFree(d_passwords);
    cudaFree(d_chaotic_values);
    throw std::runtime_error("Kernel synchronization failed");
  }

  cudaFree(d_passwords);
  cudaFree(d_chaotic_values);

  return d_indices;
};

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
                                      Image_dimensions img_dimensions,
                                      size_t block_size);

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
__host__ void block_phase_permutation_simple(unsigned char *d_image,
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
 * @brief Converts password bits into real-valued seeds for chaotic maps.
 *
 * This function takes a segment of the password (bytes) and converts it into
 * floating-point seeds (normalized to [0, 1]) that are used to initialize
 * the chaotic maps.
 *
 * @tparam T The floating-point type for the seeds (float or double).
 * @param password_segment A vector of bytes representing the password segment.
 * @param d_seeds Pointer to the device memory where the generated seeds will be
 * stored. The function allocates this memory.
 */
template <typename T>
__host__ void
convert_bits_to_real(const std::vector<unsigned char> &password_segment,
                     T **d_seeds) {

  size_t total_bytes = password_segment.size();
  size_t element_size = sizeof(unsigned int);

  if (total_bytes % element_size != 0) {
    throw std::runtime_error("Invalid length.");
  }

  size_t num_elements = total_bytes / element_size;

  cudaError_t err = cudaMalloc((void **)d_seeds, total_bytes);
  if (err != cudaSuccess)
    throw std::runtime_error("Error cudaMalloc");

  err = cudaMemcpy(*d_seeds, password_segment.data(), total_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(*d_seeds);
    throw std::runtime_error("Error cudaMemcpy");
  }

  const int threadsPerBlock = 256;
  const int gridOfBlocks =
      (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  convert_bits_to_real_kernel<T>
      <<<gridOfBlocks, threadsPerBlock>>>(*d_seeds, num_elements);

  if (cudaGetLastError() != cudaSuccess) {
    cudaFree(*d_seeds);
    throw std::runtime_error("Error en kernel convert_bits_to_real");
  }

  cudaDeviceSynchronize();
}

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
 * @brief Generate the flow stream stage using provided seeds and chaotic
 * parameter.
 * @param d_flow Device pointer to the output flow.
 * @param d_flow Device pointer to the seeds.
 * @param seeds Flow seeds per block.
 * @param cols Image width in blocks or pixels depending on the pipeline.
 * @param rows Image height in blocks or pixels depending on the pipeline.
 * @param r Chaotic map parameter.
 */
template <typename T>
__host__ void generate_flow_stream(D_pointers &d_pointers,
                                   Image_dimensions img_dimensions, T r,
                                   size_t transition_length) {

  // Launch flow stream kernel
  dim3 threadsPerBlock(256);
  dim3 numBlocks((img_dimensions.cols + threadsPerBlock.x - 1) /
                 threadsPerBlock.x);
  keystream_generation<<<numBlocks, threadsPerBlock>>>(
      d_pointers.d_flow, d_pointers.d_seeds, img_dimensions, r,
      transition_length // transition_length
  );
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("generate_flow_stream: Flow generation error");
  }
  cudaDeviceSynchronize();
}

/**
 * @brief Generates the flow keystream in parallel using a Coupled Map Lattice
 * (CML).
 *
 * This function generates a keystream based on a chaotic map. It uses a
 * parallel approach where each row of the image is processed by a separate
 * thread block (or set of blocks), with coupling between adjacent cells to
 * ensure diffusion.
 *
 * @tparam T The floating-point type for the chaotic map (float or double).
 * @param d_pointers Struct containing device pointers, specifically d_flow
 * (output) and d_seeds (input/state).
 * @param img_dimensions Struct containing the image dimensions.
 * @param r The chaotic parameter for the map.
 * @param transition_length The number of transition iterations to perform
 * before generating the stream.
 */
template <typename T>
__host__ void generate_flow_stream_parallel(D_pointers &d_pointers,
                                            Image_dimensions img_dimensions,
                                            T r, size_t transition_length) {

  // Launch flow stream kernel
  dim3 threadsPerBlock(256);
  dim3 numBlocks((img_dimensions.cols + threadsPerBlock.x - 1) /
                 threadsPerBlock.x);

  // Transition
  for (size_t i = 0; i < transition_length; i++) {
    keystream_generation_parallel<<<numBlocks, threadsPerBlock>>>(
        nullptr, d_pointers.d_seeds, img_dimensions,
        r,
        i
    );
  }

  // Stream
  for (size_t i = 0; i < img_dimensions.rows; i++) {
    keystream_generation_parallel<<<numBlocks, threadsPerBlock>>>(
        d_pointers.d_flow, d_pointers.d_seeds, img_dimensions,
        r,
        i
    );
  }
  
  // Final synchronization to ensure all stream generation is done before proceeding
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      throw std::runtime_error("generate_flow_stream_parallel: Kernel launch error");
  }
  cudaDeviceSynchronize();
}

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
    const size_t block_length, bool verbose);

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

/**
 * @brief Unstacks an interleaved (BGR) image on the device into a planar format.
 * Wrapper for deinterleave_channels_kernel.
 */
__host__ void unstack_channels_gpu(unsigned char *d_interleaved,
                                   unsigned char *d_planar,
                                   int width, int height);

/**
 * @brief Stacks a planar image on the device into an interleaved (BGR) format.
 * Wrapper for interleave_channels_kernel.
 */
__host__ void stack_channels_gpu(unsigned char *d_planar,
                                 unsigned char *d_interleaved,
                                 int width, int height);

#endif // ENCRYPTION_AUX_CUH