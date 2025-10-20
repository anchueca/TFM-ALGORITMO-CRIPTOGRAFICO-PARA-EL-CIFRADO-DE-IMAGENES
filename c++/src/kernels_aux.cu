#include "../include/kernels_aux.cuh"

__global__ void split_and_concat_kernel(const unsigned char* src, unsigned char* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int src_idx = (y * width + x) * 3;

        int dst_width = width * 3;

        int dst_idx_b = y * dst_width + x;
        int dst_idx_g = y * dst_width + x + width;
        int dst_idx_r = y * dst_width + x + 2 * width;

        dst[dst_idx_b] = src[src_idx];
        dst[dst_idx_g] = src[src_idx + 1];
        dst[dst_idx_r] = src[src_idx + 2];
    }
}

__global__ void merge_and_stack_kernel(const unsigned char* src, unsigned char* dst, int dst_width, int dst_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_width && y < dst_height) {
        int src_width = dst_width * 3;

        int src_idx_b = y * src_width + x;
        int src_idx_g = y * src_width + x + dst_width;
        int src_idx_r = y * src_width + x + 2 * dst_width;

        int dst_idx = (y * dst_width + x) * 3;

        dst[dst_idx]     = src[src_idx_b]; // Escribir B
        dst[dst_idx + 1] = src[src_idx_g]; // Escribir G
        dst[dst_idx + 2] = src[src_idx_r]; // Escribir R
    }
}

__global__ void invert_permutations_kernel(
    unsigned int *permutations, 
    unsigned int *inverses, 
    size_t block_length, 
    size_t num_blocks)
{
    // The ID of the CUDA block corresponds to the permutation index.
    int permutation_id = blockIdx.x;
    int thread_id_in_block = threadIdx.x;
    int threads_per_block = blockDim.x;

    // Use a grid-stride loop to ensure all elements of the permutation are processed
    // even if block_length > threads_per_block.
    for (int i = thread_id_in_block; i < block_length; i += threads_per_block)
    {
        // Calculate the linear index for the element in the input array.
        // Example: For permutation 2 (permutation_id=2), element 5 (i=5): idx = 2 * length + 5
        size_t idx_in = permutation_id * block_length + i;

        // Read the value at the original position. This value is the new position.
        // Example: if permutations[idx_in] is 10, it means 'i' moves to position 10.
        unsigned int new_pos = permutations[idx_in];

        // Calculate the linear index for the element in the output (inverse) array.
        size_t idx_out = permutation_id * block_length + new_pos;
        
        // Write the original position 'i' to the new position.
        // The inverse mapping is: the value at 'new_pos' is the original position 'i'.
        inverses[idx_out] = i;
    }
}