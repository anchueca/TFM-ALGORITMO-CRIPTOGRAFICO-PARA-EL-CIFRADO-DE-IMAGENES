# ifndef KERNELS_AUX_CUH
# define KERNELS_AUX_CUH

__device__ void sort_indices_by_chaotic_values(double* chaotic_vals, int* indices, int length);

# endif // KERNELS_AUX_CUH