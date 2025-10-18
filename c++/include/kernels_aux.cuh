# ifndef KERNELS_AUX_CUH
# define KERNELS_AUX_CUH

// CUDA headers primero
#include <cuda_runtime.h>

// Standard headers
#include <type_traits>

template<typename T>
__device__ inline void sort_indices_by_chaotic_values(
    int base_idx,
    T* chaotic_vals,
    unsigned int* indices,
    int block_length
) {

    for (int i = 0; i < block_length - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < block_length; j++) {
            if (chaotic_vals[base_idx + j] < chaotic_vals[base_idx + min_idx]) {
                min_idx = j;
            }
        }

        if (min_idx != i) {
            // Swap chaotic values
            T temp_val = chaotic_vals[base_idx + i];
            chaotic_vals[base_idx + i] = chaotic_vals[base_idx + min_idx];
            chaotic_vals[base_idx + min_idx] = temp_val;

            // Swap corresponding indices
            int temp_idx = indices[base_idx + i];
            indices[base_idx + i] = indices[base_idx + min_idx];
            indices[base_idx + min_idx] = temp_idx;
        }
    }
};

template<typename T>
__global__ inline void sort_indices_by_chaotic_values_global(
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

__global__ void invert_permutations_kernel(
    int* permutations, int* inverses, int length);
    
# endif // KERNELS_AUX_CUH