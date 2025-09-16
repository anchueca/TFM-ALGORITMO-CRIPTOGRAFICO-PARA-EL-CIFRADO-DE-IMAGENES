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
