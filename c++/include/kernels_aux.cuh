# ifndef KERNELS_AUX_CUH
# define KERNELS_AUX_CUH

__device__ void sort_indices_by_chaotic_values(double* chaotic_vals, int* indices, int length);

__global__ void merge_and_stack_kernel(const unsigned char* src, unsigned char* dst, int dst_width, int dst_height);
__global__ void split_and_concat_kernel(const unsigned char* src, unsigned char* dst, int width, int height);

# endif // KERNELS_AUX_CUH