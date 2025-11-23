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
__host__ void encrypt_image(cv::Mat &image, const std::string &password,
                            const EncryptionParams &params, bool verbose,
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
void encryption_process(D_pointers &d_pointers, Image_dimnesions img_dimensions,
                        size_t block_size, const EncryptionParams &params, bool verbose);

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
                          Image_dimnesions img_dimensions, size_t block_size,
                          const EncryptionParams &params, bool verbose);

void permutation_encryption_process(D_pointers &d_pointers,
                                    Image_dimnesions img_dimensions,
                                    size_t block_size);

void image_permutation_unencryption_process(D_pointers &d_pointers,
                                            Image_dimnesions img_dimensions,
                                            size_t block_size);

void image_permutation_encryption_process(D_pointers &d_pointers,
                                          Image_dimnesions img_dimensions,
                                          size_t block_size);

#endif // ENCRYPTION_CUH