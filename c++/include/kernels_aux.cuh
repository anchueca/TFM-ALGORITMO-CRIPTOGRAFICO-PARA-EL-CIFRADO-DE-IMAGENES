# ifndef KERNELS_AUX_CUH
# define KERNELS_AUX_CUH

// CUDA headers primero
#include <cuda_runtime.h>

// Standard headers
#include <type_traits>

template<typename T>
__device__ void sort_indices_by_chaotic_values(
    int base_idx,
    T* chaotic_vals,
    unsigned int* indices,
    int block_length
) {
    // Insertion Sort: empieza desde el segundo elemento (i=1)
    for (int i = 1; i < block_length; i++) {
        
        // Almacena el elemento actual (la "llave") que queremos insertar
        T key_val = chaotic_vals[base_idx + i];
        unsigned int key_idx = indices[base_idx + i];
        
        // Inicializa j en el elemento *anterior* al actual
        int j = i - 1;

        // Mueve los elementos de chaotic_vals[0...i-1] que sean
        // mayores que la llave, una posición hacia adelante
        while (j >= 0 && chaotic_vals[base_idx + j] > key_val) {
            // Desplaza el valor
            chaotic_vals[base_idx + j + 1] = chaotic_vals[base_idx + j];
            // Desplaza el índice correspondiente
            indices[base_idx + j + 1] = indices[base_idx + j];
            j = j - 1;
        }
        
        // Inserta la llave (y su índice) en la posición correcta
        // (j+1) es la primera posición "vacía" o que contenía
        // un elemento menor o igual que la llave.
        chaotic_vals[base_idx + j + 1] = key_val;
        indices[base_idx + j + 1] = key_idx;
    }
}

template<typename T>
__global__ void sort_indices_by_chaotic_values_global(
    T* d_chaotic_values,
    size_t num_blocks,
    unsigned int *indices,
    size_t block_length
) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_blocks) return;
    int base_idx = idx * block_length;

    sort_indices_by_chaotic_values<T>(base_idx,d_chaotic_values, indices, block_length);
};

__global__ void merge_and_stack_kernel(
    const unsigned char* src, unsigned char* dst, int dst_width, int dst_height);

__global__ void split_and_concat_kernel(
    const unsigned char* src, unsigned char* dst, int width, int height);
/**
 * @brief Kernel to invert a batch of permutations in parallel.
 * Each CUDA block is responsible for inverting one permutation.
 * @param permutations Input array of permutations on the GPU.
 * @param inverses Output array for the inverted permutations on the GPU.
 * @param block_length The length of a single permutation.
 * @param num_blocks (Unused) The total number of permutations. The kernel deduces this from the grid size.
 */
__global__ void invert_permutations_kernel(unsigned int *d_permutations, unsigned int *inverses, size_t block_length, size_t num_blocks);

# endif // KERNELS_AUX_CUH