#include "../include/CudaPermutation.cuh"

// Macro para verificar errores de CUDA
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// --- KERNELS ---

/**
 * @brief Inicializa los buffers en GPU.
 * Copia los índices 0..n-1 y rellena el padding con FLT_MAX.
 */
__global__ void init_buffers_kernel(float* values, int* indices, int n, int padded_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx < padded_size) {
        // Inicializar índices: identidad
        indices[idx] = idx;

        // Manejo del padding para los valores
        if (idx >= n) {
            values[idx] = FLT_MAX; // Infinito para que vayan al final al ordenar
        }
        // Nota: Los valores < n ya fueron copiados mediante cudaMemcpy
    }
}

/**
 * @brief Un paso del Bitonic Sort.
 */
__global__ void bitonic_sort_step_kernel(float* values, int* indices, int j, int k, int padded_size) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    // Procesamos si ixj > i para evitar duplicidad y mantenernos en rango
    if (ixj > i && ixj < padded_size) {
        float v1 = values[i];
        float v2 = values[ixj];

        // Dirección del ordenamiento:
        // (i & k) == 0 -> Ascendente
        // (i & k) != 0 -> Descendente
        bool ascending = ((i & k) == 0);

        // Lógica de intercambio
        if ((ascending && v1 > v2) || (!ascending && v1 < v2)) {
            // Intercambiar valores (claves)
            values[i] = v2;
            values[ixj] = v1;

            // Intercambiar índices (payload)
            int temp_idx = indices[i];
            indices[i] = indices[ixj];
            indices[ixj] = temp_idx;
        }
    }
}

// --- FUNCIONES AUXILIARES ---

int next_power_of_2(int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// --- IMPLEMENTACIÓN PRINCIPAL ---

void compute_permutation_gpu(const float* h_chaotic_sequence, int* h_permutation, int n) {
    int padded_size = next_power_of_2(n);
    
    // Punteros en dispositivo
    float* d_values;
    int* d_indices;

    // 1. Asignación de memoria en GPU (Tamaño con padding)
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_values, padded_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_indices, padded_size * sizeof(int)));

    // 2. Copiar datos de entrada (solo los n datos reales)
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_chaotic_sequence, n * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Configuración de ejecución
    int threadsPerBlock = 256;
    int blocksPerGrid = (padded_size + threadsPerBlock - 1) / threadsPerBlock;

    // 4. Inicializar el padding y los índices en GPU
    init_buffers_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_values, d_indices, n, padded_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 5. Bucle Bitonic Sort
    // k determina el tamaño de la subsecuencia monotónica
    // j determina el paso de comparación (stride)
    for (int k = 2; k <= padded_size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_values, d_indices, j, k, padded_size);
        }
    }
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 6. Copiar resultados de vuelta (solo los primeros n índices)
    // Como usamos FLT_MAX para el padding, los índices originales válidos estarán al principio.
    // Sin embargo, el padding (índices >= n) estará al final. 
    // PERO: cudaMemcpy solo copia bytes contiguos.
    // Debido al ordenamiento, los indices válidos están mezclados en el array, 
    // pero como ordenamos ascendentemente y el padding es INFINITO, 
    // el padding habrá terminado en las posiciones [n ... padded_size-1].
    // Por lo tanto, los primeros 'n' elementos son seguros de copiar.

    CHECK_CUDA_ERROR(cudaMemcpy(h_permutation, d_indices, n * sizeof(int), cudaMemcpyDeviceToHost));

    // 7. Liberar memoria
    CHECK_CUDA_ERROR(cudaFree(d_values));
    CHECK_CUDA_ERROR(cudaFree(d_indices));
}


// --- KERNELS PARA BATCHED SORT ---

/**
 * @brief Copia los datos reales a un buffer con padding (potencia de 2).
 * Convierte unsigned short a int para poder usar INT_MAX como centinela de padding.
 */
__global__ void copy_to_padded_kernel(
    const unsigned short* input_vals, const unsigned int* input_idxs,
    int* padded_vals, unsigned int* padded_idxs,
    int valid_len, int padded_len, int total_blocks) 
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int total_threads = total_blocks * padded_len; // Aunque lanzamos menos hilos, calculamos índices

    if (tid < total_blocks * padded_len) {
        int block_id = tid / padded_len;
        int offset_in_block = tid % padded_len;

        int global_padded_idx = block_id * padded_len + offset_in_block;
        
        if (offset_in_block < valid_len) {
            int global_input_idx = block_id * valid_len + offset_in_block;
            padded_vals[global_padded_idx] = (int)input_vals[global_input_idx];
            padded_idxs[global_padded_idx] = input_idxs[global_input_idx];
        } else {
            // Relleno con infinito
            padded_vals[global_padded_idx] = 2147483647; // INT_MAX
            padded_idxs[global_padded_idx] = 0; // Valor dummy
        }
    }
}

/**
 * @brief Paso del Bitonic Sort aplicado a múltiples bloques independientes.
 */
__global__ void batched_bitonic_step_kernel(
    int* values, unsigned int* indices, 
    int j, int k, 
    int padded_len, int total_blocks) 
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Cada hilo maneja una pareja, necesitamos (padded_len * total_blocks) / 2 hilos
    // Calculamos coordenadas globales
    unsigned int pair_idx = tid; 
    
    // Mapeamos el ID del hilo a la posición dentro de su bloque lógico
    unsigned int idx_in_total = pair_idx; // Este no es el índice final del array
    
    // Necesitamos reconstruir i e ixj basados en la lógica bitónica PERO relativa al bloque
    int block_id = tid / (padded_len / 2);
    int tid_in_block = tid % (padded_len / 2);
    
    // Lógica Bitónica estándar aplicada localmente
    // Un hilo representa una comparación. Pero 'i' debe saltar huecos.
    // Es más fácil calcular 'i' globalmente asumiendo que ejecutamos hilos para cubrir todo.
    
    // Truco: Usamos la lógica standard pero añadimos el offset del bloque
    unsigned int i_local = tid_in_block * 2; // Mapeo ingenuo, necesitamos insertar el bit de gap si fuera necesario,
                                             // pero en la formulación iterativa global, 'i' es directo si filtramos.
    
    // Reimplementación robusta para Global Memory:
    // Ejecutamos un hilo por cada elemento, pero solo operamos si (i < ixj)
    // Esto desperdicia la mitad de los hilos pero simplifica la indexación masiva.
    // RE-MAPEO: Vamos a usar la estrategia de "1 hilo = 1 elemento" y filtrar.
    
    unsigned int i = tid;
    if (i >= total_blocks * padded_len) return;

    int my_block = i / padded_len;
    int i_local_real = i % padded_len;
    
    int ixj_local = i_local_real ^ j;
    
    // Si el compañero está fuera de mi bloque (no debería pasar si j < padded_len), abortar
    if (ixj_local >= padded_len) return;

    int ixj = my_block * padded_len + ixj_local;

    if (ixj > i) {
        int v1 = values[i];
        int v2 = values[ixj];
        
        bool ascending = ((i_local_real & k) == 0);
        
        if ((ascending && v1 > v2) || (!ascending && v1 < v2)) {
            // Swap Values
            values[i] = v2;
            values[ixj] = v1;
            
            // Swap Indices
            unsigned int tmp = indices[i];
            indices[i] = indices[ixj];
            indices[ixj] = tmp;
        }
    }
}

/**
 * @brief Copia los resultados válidos de vuelta al buffer original.
 */
__global__ void copy_from_padded_kernel(
    const int* padded_vals, const unsigned int* padded_idxs,
    unsigned int* output_idxs,
    int valid_len, int padded_len, int total_blocks)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (tid < total_blocks * valid_len) {
        int block_id = tid / valid_len;
        int offset_in_block = tid % valid_len;
        
        // Los datos válidos (no infinito) estarán al principio de cada bloque padded
        // porque ordenamos ascendentemente.
        int global_padded_idx = block_id * padded_len + offset_in_block;
        int global_output_idx = tid;
        
        output_idxs[global_output_idx] = padded_idxs[global_padded_idx];
    }
}

// Helper CPU
int next_pow2(int n) {
    if (n == 0) return 1;
    n--; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    return n + 1;
}

/**
 * @brief Función Host que orquesta el Bitonic Sort por lotes.
 */
void batched_gpu_argsort(unsigned short* d_keys, unsigned int* d_indices, 
                         size_t num_blocks, size_t block_len) {
    
    int padded_len = next_pow2(block_len);
    size_t total_padded_elements = num_blocks * padded_len;

    // 1. Allocate Temporary Padded Buffers
    int* d_padded_keys;
    unsigned int* d_padded_indices;
    
    cudaMalloc(&d_padded_keys, total_padded_elements * sizeof(int));
    cudaMalloc(&d_padded_indices, total_padded_elements * sizeof(unsigned int));

    // 2. Copy and Pad (Input -> Padded Temp)
    int threads = 256;
    int blocks = (total_padded_elements + threads - 1) / threads;
    
    copy_to_padded_kernel<<<blocks, threads>>>(
        d_keys, d_indices, 
        d_padded_keys, d_padded_indices, 
        block_len, padded_len, num_blocks
    );
    cudaDeviceSynchronize();

    // 3. Batched Bitonic Sort Loop
    // Nota: Lanzamos hilos para cubrir todos los elementos PADDED
    blocks = (total_padded_elements + threads - 1) / threads;

    for (int k = 2; k <= padded_len; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            batched_bitonic_step_kernel<<<blocks, threads>>>(
                d_padded_keys, d_padded_indices, 
                j, k, 
                padded_len, num_blocks
            );
        }
    }
    cudaDeviceSynchronize();

    // 4. Copy Back (Padded Temp -> Output)
    // Solo necesitamos hilos para los elementos reales
    int total_real_elements = num_blocks * block_len;
    blocks = (total_real_elements + threads - 1) / threads;
    
    copy_from_padded_kernel<<<blocks, threads>>>(
        d_padded_keys, d_padded_indices,
        d_indices, // Escribimos el resultado final aquí
        block_len, padded_len, num_blocks
    );
    cudaGetLastError(); // Check errors

    // 5. Cleanup
    cudaFree(d_padded_keys);
    cudaFree(d_padded_indices);
}