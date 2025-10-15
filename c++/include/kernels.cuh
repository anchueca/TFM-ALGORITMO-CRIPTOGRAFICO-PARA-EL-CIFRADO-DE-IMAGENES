# ifndef KERNELS_CUH
# define KERNELS_CUH

// CUDA headers primero
#include <cuda_runtime.h>

// Standard headers después
#include <iostream>
#include <vector>

// Project headers al final
#include "kernels_aux.cuh"
#include "automata.cuh"

__device__ double uno(double x, double r);

__global__ void flow_encrypt_recursive(
    unsigned char *image,
    unsigned char *image_out,
    const unsigned char *seeds,
    int width,
    int height,
    double r,
    int rounds);

__global__ void permute_blocks_kernel(
    unsigned char *image, unsigned char *image_out, unsigned int *permutations,
    size_t block_size, size_t cols, size_t rows);

__global__ void generate_chaotic(
    unsigned char* passwords, size_t num_blocks, double* chaotic_vals,
    unsigned int* indices, double r, size_t block_length, size_t transition_length);

__global__ void permute_columns_kernel(
    unsigned char *image, unsigned char *image_out, unsigned int *permutation, size_t cols, size_t rows);

__global__ void permute_rows_kernel(
    unsigned char *image, unsigned char *image_out, unsigned int *permutation, size_t cols, size_t rows);

__global__ void generate_automata_chaotic(unsigned int** automata_states, unsigned int* d_chaotic_values, size_t num_blocks, unsigned int *indices, size_t block_length);

/*template <typename T>
__device__ void convert_to_bitstream(size_t size, unsigned int* d_state, T* bitstream) {
        // Determina cuántos elementos de tipo 'T' caben en la memoria
        size_t num_elements = (size + sizeof(T) * 8 - 1) / (sizeof(T) * 8);
        size_t size_in_bytes = num_elements * sizeof(T);
        
        // Copia los datos del tipo genérico (T) a la representación en bytes
        cudaMemcpy(bitstream, d_state, size_in_bytes, cudaMemcpyDeviceToHost);
    }*/

# endif // KERNELS_CUH