# ifndef ENCRYPTION_CUH
# define ENCRYPTION_CUH

// CUDA headers primero
#include <cuda_runtime.h>

// Standard headers
#include <algorithm>
#include <vector>
#include <iostream>

// Project headers
#include "kernels.cuh"
#include "aux.cuh"
#include "automata.cuh"
#include "encryption_aux.cuh"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;

void encrypt_image(cv::Mat image, const std::string& password, int rounds, int verbose, bool encrypt);
void encryption_process(unsigned char** d_image, unsigned char** d_image_out, unsigned int* d_permutation_rows, unsigned int* d_permutation_cols, unsigned int* d_permutation_blocks, size_t cols, size_t rows, std::vector<unsigned char> flow_seeds, size_t block_size, size_t rounds);
void unencryption_process(unsigned char** d_image, unsigned char** d_image_out, unsigned int* d_permutation_rows, unsigned int* d_permutation_cols, unsigned int* d_permutation_blocks, size_t cols, size_t rows, std::vector<unsigned char> flow_seeds, size_t block_size, size_t rounds);


# endif // ENCRYPTION_CUH