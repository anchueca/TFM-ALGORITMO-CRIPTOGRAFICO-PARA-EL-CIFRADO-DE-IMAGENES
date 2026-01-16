#ifndef AUX_CUH
#define AUX_CUH

#include <istream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openssl/evp.h>
#include <vector>

#include <iostream>

#include "kernels_aux.cuh"
#include "structs.cuh"

/**
 * @brief Generates a SHA-256 hash of the input string.
 *
 * This function computes the SHA-256 hash of the provided input string and
 * returns a vector of bytes containing the hash. If the requested length is
 * less than the hash size, the result is truncated.
 *
 * @param input The input string to be hashed.
 * @param length The desired length of the output hash in bytes.
 * @return A vector of unsigned characters containing the hash.
 */
__host__ std::vector<unsigned char> generate_hash(const std::string &input,
                                                  size_t length);

/**
 * @brief Generates a SHA-3 (SHAKE256) hash of the input buffer.
 *
 * @param input Pointer to the input data.
 * @param input_len Length of the input data in bytes.
 * @param length The desired length of the output hash in bytes.
 * @return A vector of unsigned characters containing the hash.
 */
__host__ std::vector<unsigned char>
generate_hash(const unsigned char *input, size_t input_len, size_t length);

/**
 * @brief Calculate password segments from a textual password.
 *
 * This function derives multiple password segments used across blocks and
 * automata. Parameters control how many blocks are generated, number of rounds,
 * and target image dimensions.
 *
 * @param input Password string provided by the user.
 * @param num_blocks Number of blocks to split the password into.
 * @param image_height Height of target image (for sizing segments).
 * @param image_width Width of target image (for sizing segments).
 * @return Vector of password byte vectors, one per block.
 */
__host__ std::vector<std::vector<unsigned char>>
calculate_password(const std::string &input, Image_dimensions img_dimensions,
                   bool verbose, bool use_raw_key);

/**
 * @brief Unstacks a 3-channel image into a single-channel wide image.
 *
 * @param image Input image.
 * @param verbose Enable verbose logging.
 * @return cv::Mat Single-channel image.
 */
cv::Mat unstack_channels(const cv::Mat &image, bool verbose);

/**
 * @brief Stacks a single-channel wide image back into a 3-channel image if
 * needed.
 *
 * @param image Output image (modified in place).
 * @param processed_image Input single-channel wide image.
 * @param is_color Whether the original image was color.
 * @param verbose Enable verbose logging.
 */
void stack_channels(cv::Mat &image, const cv::Mat &processed_image,
                    bool is_color, bool verbose);

/**
 * @brief Calculates a hash of the image data using SHAKE256.
 *
 * @param image Input image.
 * @param length Desired hash length in bytes.
 * @return unsigned short 16-bit hash (truncated if length > 2).
 */
unsigned short calculate_image_hash(const cv::Mat &image, size_t length);

/**
 * @brief Extracts a hidden message (image hash) using EXIF metadata.
 *
 * @param image The image with hidden info.
 * @param stego_key The password segment for steganography.
 * @param input_path Path to the image file to read EXIF.
 * @return unsigned short The extracted 16-bit hash.
 */
unsigned short extract_message_caos(cv::Mat &image,
                                    const std::vector<unsigned char> &stego_key,
                                    const std::string &input_path,
                                    const std::string &exif_hex = "");

/**
 * @brief Embeds a message (image hash) and stores recovery info in EXIF.
 *
 * @param image The image to modify.
 * @param image_hash The 16-bit hash to hide.
 * @param stego_key The password segment for steganography.
 * @param output_path Path to save the image (for EXIF).
 */
void embed_message_caos(cv::Mat &image, unsigned short image_hash,
                        const std::vector<unsigned char> &stego_key,
                        const std::string &output_path);

/**
 * @brief Pads the image to a square with dimensions multiple of blockSize.
 *
 * @param input Original image.
 * @param blockSize Block size.
 * @param original_channels Number of channels of the original image before
 * unstacking.
 * @return cv::Mat Padded square image.
 */
cv::Mat padImageToSquare(const cv::Mat &input, int blockSize,
                         int original_channels);

/**
 * @brief Reverts the padding process by extracting the original image.
 *
 * @param squared Padded square image.
 * @param out_original_channels Pointer to return the number of original
 * channels.
 * @return cv::Mat Original image.
 */
cv::Mat unpadFromSquare(const cv::Mat &squared, int *out_original_channels);

#endif // AUX_CUH