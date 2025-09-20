# ifndef AUX_CUH
# define AUX_CUH

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <openssl/evp.h>

#include "kernels_aux.cuh"

__host__ cv::Mat unstack_image(cv::Mat image);

__host__ cv::Mat stack_image(cv::Mat image);

__host__ std::vector<unsigned char> generate_sha3_hash(const std::string &input, size_t length);

__host__ std::vector<std::vector<unsigned char>> calculate_password(const std::string &input, int num_blocks, int precision_level, int rounds, int image_height, int image_width);

# endif // AUX_CUH