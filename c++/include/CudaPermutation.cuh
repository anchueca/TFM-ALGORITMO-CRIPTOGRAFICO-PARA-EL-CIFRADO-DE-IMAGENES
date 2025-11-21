#ifndef CUDA_PERMUTATION_H
#define CUDA_PERMUTATION_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <chrono>

/**
 * @brief Genera una permutación (argsort) basada en una secuencia caótica utilizando la GPU.
 * * Realiza un Bitonic Sort paralelo. Maneja automáticamente el padding si n no es potencia de 2.
 * * @param h_chaotic_sequence Puntero al array de floats con los valores caóticos (en CPU).
 * @param h_permutation Puntero al array de ints donde se guardará el resultado (en CPU).
 * @param n Número de elementos.
 */
void compute_permutation_gpu(const float* h_chaotic_sequence, int* h_permutation, int n);
__global__ void init_buffers_kernel(float* values, int* indices, int n, int padded_size);
__global__ void bitonic_sort_step_kernel(float* values, int* indices, int j, int k, int padded_size);
void compute_permutation_gpu(const float* h_chaotic_sequence, int* h_permutation, int n);
/**
 * @brief Copia los datos reales a un buffer con padding (potencia de 2).
 * Convierte unsigned short a int para poder usar INT_MAX como centinela de padding.
 */
__global__ void copy_to_padded_kernel(
    const unsigned short* input_vals, const unsigned int* input_idxs,
    int* padded_vals, unsigned int* padded_idxs,
    int valid_len, int padded_len, int total_blocks);

    __global__ void batched_bitonic_step_kernel(
    int* values, unsigned int* indices, 
    int j, int k, 
    int padded_len, int total_blocks) ;

    __global__ void copy_from_padded_kernel(
    const int* padded_vals, const unsigned int* padded_idxs,
    unsigned int* output_idxs,
    int valid_len, int padded_len, int total_blocks);

    int next_pow2(int n);

    void batched_gpu_argsort(unsigned short* d_keys, unsigned int* d_indices, 
                         size_t num_blocks, size_t block_len) ;
#endif // CUDA_PERMUTATION_H