#ifndef AUX_CUH
#define AUX_CUH

#include <istream>
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
 * @brief Calculate password segments from a textual password.
 *
 * This function derives multiple password segments used across blocks and
 * automata. Parameters control how many blocks are generated, the precision
 * level of derivation, number of rounds, and target image dimensions.
 *
 * @param input Password string provided by the user.
 * @param num_blocks Number of blocks to split the password into.
 * @param precision_level Precision level used during derivation.
 * @param rounds Number of internal rounds for the KDF.
 * @param image_height Height of target image (for sizing segments).
 * @param image_width Width of target image (for sizing segments).
 * @return Vector of password byte vectors, one per block.
 */
__host__ std::vector<std::vector<unsigned char>>
calculate_password(const std::string &input, size_t num_blocks,
                   size_t precision_level, Image_dimensions img_dimensions,
                   bool verbose);

#endif // AUX_CUH