# ifndef ENCRYPTION_CUH
# define ENCRYPTION_CUH

#include <algorithm>
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "kernels.cuh"
#include "aux.cuh"

using namespace std;

void encrypt_image(cv::Mat image, const std::string& password, int rounds, int verbose);

__host__ std::vector<std::vector<int>> generate_permutations(
    const std::vector<unsigned char> block_passwords, size_t block_length,
    size_t num_blocks);

__host__ void block_phase_permutation(
    unsigned char* d_image, unsigned char* d_image_out, std::vector<std::vector<int>> &block_permutations, size_t cols, size_t rows, size_t block_size);

__host__ void rows_and_columns_permutation(
    unsigned char* d_image, unsigned char* d_image_out, std::vector<int> &row_permutation, std::vector<int> &col_permutation, size_t cols, size_t rows, bool inverse);

# endif // ENCRYPTION_CUH