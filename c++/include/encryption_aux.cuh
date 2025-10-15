# ifndef ENCRYPTION_AUX_CUH
# define ENCRYPTION_AUX_CUH

#include <algorithm>
#include <vector>
#include <iostream>

#include "kernels.cuh"
#include "automata.cuh"

using namespace std;

__host__ unsigned int* generate_flow_permutations(
    const std::vector<unsigned char> block_passwords, size_t block_length,
    size_t num_blocks);

__host__ void block_phase_permutation(
    unsigned char* d_image, unsigned char* d_image_out, unsigned int* block_permutations, size_t cols, size_t rows, size_t block_size);

__host__ void rows_and_columns_permutation(
    unsigned char* d_image, unsigned char* d_image_out, unsigned int *row_permutation, unsigned int *col_permutation, size_t cols, size_t rows, bool inverse);

__host__ void flow_encrypt(
    unsigned char *image,
    unsigned char *image_out,
    const std::vector<unsigned char> seeds,
    size_t cols,
    size_t rows,
    double r,
    int rounds);

__host__ unsigned int* generate_automata_permutations(const std::vector<ElementalCelularAutomata*> automatas, size_t block_length, size_t num_blocks);

# endif // ENCRYPTION_AUX_CUH