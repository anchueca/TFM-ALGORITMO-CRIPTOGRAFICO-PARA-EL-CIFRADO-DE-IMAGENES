# ifndef AUX_HPP
# define AUX_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <openssl/evp.h>

void unstack_image(cv::Mat& image);

void stack_image(cv::Mat& concatenated);

std::vector<unsigned char> generate_sha3_hash(const std::string &input, size_t length);

std::vector<std::vector<unsigned char>> calculate_password(const std::string &input, int num_blocks, int precision_level, int rounds, int image_height, int image_width);

# endif // AUX_HPP