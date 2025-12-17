#ifndef ENCRYPTION_CUH
#define ENCRYPTION_CUH

// CUDA headers
#include <cuda_runtime.h>

// Standard headers
#include <algorithm>
#include <algorithm> // For std::swap
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

// Project headers
#include "automata.cuh"
#include "aux.cuh"
#include "encryption_aux.cuh"
#include "kernels.cuh"
#include "structs.cuh"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;

/**
 * @brief Orchestrates the high-level image encryption process.
 * * Logic for Color Images:
 * - Pre-processing: Unstacks RGB into a wide single-channel matrix ([R][G][B])
 * for GPU processing.
 * - Post-processing: Stacks the wide matrix back into RGB ([R,G,B]).
 * * This ensures that both the Encrypted result and the Decrypted result
 * maintain the original format (e.g., 3 channels).
 * * @param image Reference to the input/output OpenCV matrix. Modified in
 * place.
 * @param password The user-provided password for key generation.
 * @param params Struct containing configuration for encryption (block size,
 * rounds, etc.).
 * @param verbose Flag to enable console logging for performance metrics.
 * @param encrypt True to encrypt, False to decrypt.
 */
__host__ void encrypt_image(cv::Mat &image, std::vector<std::vector<unsigned char>> &password,
                            const Image_dimensions &img_dimensions, const EncryptionParams &params, bool verbose,
                            bool encrypt);

/**
 * @brief Internal pipeline function that performs the encryption stages
 * (in-place on device buffers).
 *
 * @param d_image Pointer to device pointer of the current image buffer.
 * @param d_image_out Pointer to device pointer for the output image buffer.
 * @param d_permutation_rows Device pointer to row permutations.
 * @param d_permutation_cols Device pointer to column permutations.
 * @param d_permutation_blocks Device pointer to block permutations.
 * @param img_dimensions Strcut with the number of columns and rows (image width
 * and image height) in pixels or blocks.
 * @param flow_seeds Seeds used by the flow generator.
 * @param block_size Block size used for block permutations.
 * @param rounds Number of rounds for this stage.
 * @param verbose Print verbose info if true.
 */
void encryption_process(D_pointers &d_pointers, Image_dimensions img_dimensions,
                        size_t block_size, const EncryptionParams &params,
                        bool verbose);

/**
 * @brief Internal pipeline function that performs the decryption (inverse of
 * encryption_process).
 *
 * @param d_image Pointer to device pointer of the current image buffer.
 * @param d_image_out Pointer to device pointer for the output image buffer.
 * @param d_permutation_rows Device pointer to row permutations.
 * @param d_permutation_cols Device pointer to column permutations.
 * @param d_permutation_blocks Device pointer to block permutations.
 * @param img_dimensions Strcut with the number of columns and rows (image width
 * and image height) in pixels or blocks.
 * @param flow_seeds Seeds used by the flow generator.
 * @param block_size Block size used for block permutations.
 * @param rounds Number of rounds for this stage.
 */
void unencryption_process(D_pointers &d_pointers,
                          Image_dimensions img_dimensions, size_t block_size,
                          const EncryptionParams &params, bool verbose);

/**
 * @brief Executes the permutation-only encryption process.
 *
 * This function handles the encryption pipeline when only permutations are
 * applied (without the diffusion/XOR stage). It orchestrates the block, row,
 * and column permutations.
 *
 * @param d_pointers Struct containing device pointers for image data and
 * permutations.
 * @param img_dimensions Struct containing the image dimensions.
 * @param block_size The size of the blocks used for permutation.
 */
void permutation_encryption_process(D_pointers &d_pointers,
                                    Image_dimensions img_dimensions,
                                    size_t block_size);

void warmup_gpu();

/**
 * @brief Executes the inverse permutation process for decryption.
 *
 * This function reverses the permutation steps applied during encryption. It
 * applies the inverse row, column, and block permutations to restore the
 * original image structure.
 *
 * @param d_pointers Struct containing device pointers for image data and
 * inverse permutations.
 * @param img_dimensions Struct containing the image dimensions.
 * @param block_size The size of the blocks used for permutation.
 */
void image_permutation_unencryption_process(D_pointers &d_pointers,
                                            Image_dimensions img_dimensions,
                                            size_t block_size);

/**
 * @brief Executes the forward permutation process for encryption.
 *
 * This function applies the row, column, and block permutations to the image
 * data as part of the encryption process.
 *
 * @param d_pointers Struct containing device pointers for image data and
 * permutations.
 * @param img_dimensions Struct containing the image dimensions.
 * @param block_size The size of the blocks used for permutation.
 */
void image_permutation_encryption_process(D_pointers &d_pointers,
                                          Image_dimensions img_dimensions,
                                          size_t block_size);

/**
 * @brief Helper functions for the encryption pipeline.
 */
void print_encryption_report(const cv::Mat &image, const Image_dimensions &img_dimensions,
                              const EncryptionParams &params, bool encrypt);

void setup_permutations(D_pointers &d_pointers, std::vector<std::vector<unsigned char>> &password,
                        const Image_dimensions &img_dimensions, const EncryptionParams &params, bool verbose);

void allocate_and_transfer_image(D_pointers &d_pointers, cv::Mat &image, const EncryptionParams &params);

void transfer_back_and_cleanup(D_pointers &d_pointers, cv::Mat &image);

#endif // ENCRYPTION_CUH