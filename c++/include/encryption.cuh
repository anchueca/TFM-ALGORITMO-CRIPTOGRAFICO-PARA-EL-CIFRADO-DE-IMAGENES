#ifndef ENCRYPTION_CUH
#define ENCRYPTION_CUH

// CUDA headers
#include <cuda_runtime.h>

// Standard headers
#include <algorithm>
#include <chrono>
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
 * @brief Top-level helper to encrypt or decrypt an image using the provided
 * password and parameters.
 *
 * If encrypt is true the function encrypts, otherwise it attempts to decrypt
 * using the inverse operations. The function is a convenience wrapper that sets
 * up GPU buffers and coordinates the stages of the pipeline.
 *
 * @param image Input image (cv::Mat) to encrypt/decrypt.
 * @param password Password string used for key derivation.
 * @param params EncryptionParams struct controlling algorithm behavior.
 * @param verbose If true, prints progress and debug information.
 * @param encrypt Whether to run encryption (true) or decryption (false).
 */
void encrypt_image(cv::Mat image, const std::string &password,
                   const EncryptionParams &params, bool verbose, bool encrypt);

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
                        std::vector<unsigned char> flow_seeds,
                        size_t block_size, size_t rounds, bool verbose);

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
                          Image_dimnesions img_dimensions,
                          std::vector<unsigned char> flow_seeds,
                          size_t block_size, size_t rounds);

void permutation_unencryption_process(D_pointers &d_pointers,
                                      Image_dimnesions img_dimensions,
                                      size_t block_size);

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