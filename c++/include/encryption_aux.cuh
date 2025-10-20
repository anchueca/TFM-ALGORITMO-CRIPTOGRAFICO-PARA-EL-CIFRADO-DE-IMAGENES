# ifndef ENCRYPTION_AUX_CUH
# define ENCRYPTION_AUX_CUH

#include <algorithm>
#include <vector>
#include <iostream>

#include "kernels.cuh"
#include "automata.cuh"

__host__ unsigned int* generate_flow_permutations(
    const std::vector<unsigned char> block_passwords, size_t block_length,
    size_t num_blocks);

__host__ void block_phase_permutation(
    unsigned char* d_image, unsigned char* d_image_out, unsigned int* block_permutations, size_t cols, size_t rows, size_t block_size);

__host__ void rows_and_columns_permutation(
    unsigned char* d_image, unsigned char* d_image_out, unsigned int *d_row_permutations, unsigned int *d_col_permutations, size_t cols, size_t rows, bool inverse);

__host__ void flow_encrypt(
    unsigned char *image,
    unsigned char *image_out,
    const std::vector<unsigned char> seeds,
    size_t cols,
    size_t rows,
    double r,
    int rounds);

__host__ unsigned int* generate_automata_permutations(const std::vector<ElementalCelularAutomata*> automatas, const size_t steps, const size_t block_length);

/**
 * @brief Inverts a batch of permutations stored on the GPU.
 * @param d_permutations Device pointer to the input array containing N contiguous permutations.
 * @param block_length The length of a single permutation.
 * @param num_permutations The total number of permutations in the batch.
 * @return A new device pointer to the array of inverted permutations. The caller is responsible for freeing this memory.
 */
__host__ void inverse_permutations(unsigned int** d_permutations, size_t block_length, size_t num_blocks);

# endif // ENCRYPTION_AUX_CUH