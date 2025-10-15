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

__global__ void invert_permutations_kernel(int *permutations, int *inverses, int length)
{
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int threads_per_block = blockDim.x;

    for (int i = thread_id; i < length; i += threads_per_block)
    {
        int idx = block_id * length + i;
        int pos = permutations[idx];
        inverses[block_id * length + pos] = i;
    }
}