#ifndef STEGANOGRAPHY_HPP
#define STEGANOGRAPHY_HPP

#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

/**
 * @brief Embeds a message into an image using chaos-based steganography.
 * @param image The image to modify (LSBs will be changed).
 * @param message Sequence of bits to hide.
 * @param key Password bits used to generate the chaotic sequence.
 * @return Sequence of bits (R) needed for lossless recovery of the original
 * image.
 */
std::vector<bool> embed_message_caos(cv::Mat &image,
                                     const std::vector<bool> &message,
                                     const std::vector<bool> &key);

/**
 * @brief Embeds a message and stores recovery information in EXIF metadata.
 * @param image The image to modify (LSBs will be changed).
 * @param message Sequence of bits to hide.
 * @param key Password bits used to generate the chaotic sequence.
 * @param output_path Path where the image will be saved (for EXIF metadata).
 * @return Sequence of bits (R) for recovery (also stored in EXIF).
 */
std::vector<bool> embed_message_caos_with_exif(cv::Mat &image,
                                               const std::vector<bool> &message,
                                               const std::vector<bool> &key,
                                               const std::string &output_path);

/**
 * @brief Extracts a hidden message and restores the original image.
 * @param image The image with hidden information.
 * @param recovery_info Sequence of bits (R) used to restore original LSBs.
 * @param key Password bits used to locate the hidden bits.
 * @return The extracted sequence of bits.
 */
std::vector<bool> extract_message_caos(cv::Mat &image,
                                       const std::vector<bool> &recovery_info,
                                       const std::vector<bool> &key);

/**
 * @brief Extracts a hidden message using recovery info from EXIF metadata.
 * @param image The image with hidden information.
 * @param key Password bits used to locate the hidden bits.
 * @param input_path Path to the image file (for reading EXIF metadata).
 * @return The extracted sequence of bits.
 */
std::vector<bool> extract_message_caos_with_exif(cv::Mat &image,
                                                 const std::vector<bool> &key,
                                                 const std::string &input_path);

#endif
