#include "../include/kernels_aux.cuh"

__device__ void sort_indices_by_chaotic_values(double* chaotic_vals, int* indices, int length) {
    // Usamos el algoritmo de ordenación por burbuja (Bubble Sort)
    for (int i = 0; i < length - 1; i++) {
        for (int j = 0; j < length - 1 - i; j++) {
            // Si el valor caótico actual es mayor que el siguiente, intercambiamos
            if (chaotic_vals[j] > chaotic_vals[j + 1]) {
                // Intercambiar los valores de chaotic_vals
                double temp = chaotic_vals[j];
                chaotic_vals[j] = chaotic_vals[j + 1];
                chaotic_vals[j + 1] = temp;

                // Intercambiar los índices correspondientes
                int temp_index = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp_index;
            }
        }
    }
}


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