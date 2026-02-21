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
 *
 * The image received has already been unstacked and padded on the CPU.
 * This function performs the GPU-based encryption/decryption operations
 * and returns the processed image in the same unstacked format.
 *
 * @param image Reference to the input/output OpenCV matrix (already unstacked
 * and padded). Modified in place.
 * @param password The user-provided password for key generation.
 * @param img_dimensions Dimensions of the padded image.
 * @param params Struct containing configuration for encryption (block size,
 * rounds, etc.).
 * @param verbose Flag to enable console logging for performance metrics.
 * @param encrypt True to encrypt, False to decrypt.
 */
__host__ void encrypt_image(cv::Mat &image,
                            std::vector<std::vector<unsigned char>> &password,
                            const Image_dimensions &img_dimensions,
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

void warmup_gpu();

/**
 * @brief Helper functions for the encryption pipeline.
 */
void print_encryption_report(const cv::Mat &image,
                             const Image_dimensions &img_dimensions,
                             const EncryptionParams &params, bool encrypt);

/**
 * @brief Sets up the permutations for encryption or decryption.
 *
 * This function generates the row and column permutations based on the provided
 * password and image dimensions.
 *
 * @param d_pointers Struct containing device pointers for image data and
 * permutations.
 * @param password The user-provided password for key generation.
 * @param img_dimensions Struct containing the image dimensions.
 * @param params Struct containing configuration for encryption (block size,
 * rounds, etc.).
 * @param verbose Print verbose info if true.
 */
void setup_permutations(D_pointers &d_pointers,
                        std::vector<std::vector<unsigned char>> &password,
                        const Image_dimensions &img_dimensions,
                        const EncryptionParams &params, bool verbose);

/**
 * @brief Allocates GPU memory and transfers the already unstacked and padded
 * image.
 *
 * @param d_pointers Struct containing device pointers.
 * @param image The image (already unstacked and padded on CPU).
 * @param params Encryption parameters.
 */
void allocate_and_transfer_image(D_pointers &d_pointers, cv::Mat &image,
                                 const EncryptionParams &params, bool verbose);

/**
 * @brief Transfers the processed image back from GPU and frees GPU memory.
 *
 * The image returned is still in unstacked format. Stacking and unpadding
 * are handled on CPU after this function returns.
 *
 * @param d_pointers Struct containing device pointers.
 * @param image The output image buffer (unstacked format).
 */
void transfer_back_and_cleanup(D_pointers &d_pointers, cv::Mat &image);

#endif // ENCRYPTION_CUH