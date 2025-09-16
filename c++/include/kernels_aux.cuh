# ifndef KERNELS_AUX_CUH
# define KERNELS_AUX_CUH

// Ordena los índices según los valores caóticos usando un algoritmo simple.
// Se define como inline __device__ para que la función esté disponible
// en cada unidad de traducción que incluya este header y evitar problemas
// de enlace de código de dispositivo entre archivos .cu.
__device__ void sort_indices_by_chaotic_values(double* chaotic_vals, int* indices, int length);

# endif // KERNELS_AUX_CUH