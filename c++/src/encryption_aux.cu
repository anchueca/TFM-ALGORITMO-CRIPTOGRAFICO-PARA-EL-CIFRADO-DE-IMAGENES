#include "../include/encryption_aux.cuh"

/// @brief Generate a set of permutations based on chaotic sequences derived from block passwords.
/// @param block_passwords Vector de contraseñas de los bloques.
/// @param block_length La longitud de los bloques.
/// @param num_blocks El número de bloques.
/// @return Un vector de vectores de enteros que representa las permutaciones generadas.
__host__ unsigned int* generate_flow_permutations(const std::vector<unsigned char> block_passwords, size_t block_length, size_t num_blocks)
{
    if (block_passwords.size() < num_blocks) {
        throw std::runtime_error("Insufficient passwords for blocks");
    }
    
    size_t total_size = num_blocks * block_length;
    
    unsigned char *d_passwords = nullptr;
    unsigned int *d_indices = nullptr;
    double *d_chaotic_values = nullptr;
    
    cudaError_t err = cudaMalloc(&d_passwords, num_blocks * sizeof(unsigned char));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for passwords");
    }
    
    err = cudaMalloc(&d_indices, total_size * sizeof(unsigned int));
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        throw std::runtime_error("Failed to allocate device memory for indices");
    }
    
    err = cudaMalloc(&d_chaotic_values, total_size * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_indices);
        throw std::runtime_error("Failed to allocate device memory for chaotic values");
    }
    
    // Copy
    err = cudaMemcpy(d_passwords, block_passwords.data(), num_blocks * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_indices);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("Failed to copy passwords to device");
    }
    
    const int threadsPerBlock = 256;
    const int numBlocks = (num_blocks + threadsPerBlock - 1) / threadsPerBlock;
    double r = 0.998;
    int transition_length = 20;
    
    generate_chaotic<<<numBlocks, threadsPerBlock>>>(
        d_passwords, num_blocks, d_chaotic_values, d_indices, r, block_length, transition_length
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("Kernel execution failed");
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("Kernel synchronization failed");
    }
    
    cudaFree(d_passwords);
    cudaFree(d_chaotic_values);
    
    return d_indices;
}

/// @brief Generate a set of permutations based on elemental automata sequences
/// @param automatas 
/// @param block_length La longitud de los bloques.
/// @param num_blocks El número de bloques.
/// @return Un puntero 
__host__ unsigned int* generate_automata_permutations(const std::vector<ElementalCelularAutomata*> automatas, const size_t steps, const size_t block_length)
{

    size_t num_blocks = automatas.size();
    size_t total_size = num_blocks * block_length;

    if (automatas[0]->get_size()*num_blocks != total_size * 16)
        throw std::runtime_error(
            "Incompatible automata size ("
            + std::to_string(automatas[0]->get_size() * num_blocks) // Tamaño total de los autómatas
            + ") and block length ("
            + std::to_string(total_size * 16) // Tamaño del bloque en bytes
            + ")"
        );

    //iterate
    for(int i=0; i< num_blocks;i++){
        automatas[i]->iterate(steps);
    }

    
    unsigned int **d_automatas = nullptr; //Array of pointers to automata states
    unsigned int *d_indices = nullptr;
    unsigned short *d_chaotic_values = nullptr; // Here the automata states are gonna be copied
    
    cudaError_t err = cudaMalloc(&d_automatas, num_blocks * sizeof(unsigned int*));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for passwords");
    }
    
    err = cudaMalloc(&d_indices, total_size * sizeof(unsigned int));
    if (err != cudaSuccess) {
        cudaFree(d_automatas);
        throw std::runtime_error("Failed to allocate device memory for indices");
    }
    
    err = cudaMalloc(&d_chaotic_values, total_size * sizeof(unsigned short));
    if (err != cudaSuccess) {
        cudaFree(d_automatas);
        cudaFree(d_indices);
        throw std::runtime_error("Failed to allocate device memory for chaotic values");
    }
    
    cudaDeviceSynchronize();

    const unsigned int **pointers_to_automata_states = new const unsigned int *[num_blocks];
    for(int i=0; i< num_blocks;i++){
        pointers_to_automata_states[i]=automatas[i]->get_cuda_state();
    }

    err = cudaMemcpy(d_automatas,pointers_to_automata_states, num_blocks * sizeof(unsigned int*),cudaMemcpyHostToDevice);
    delete[] pointers_to_automata_states;
    if (err != cudaSuccess) {
        cudaFree(d_automatas);
        cudaFree(d_indices);
        throw std::runtime_error("Failed to copy device memory for automatas states");
    }
    
    const int threadsPerBlock = 256;
    const int numKerBlocksChaotic = (num_blocks*block_length + threadsPerBlock - 1) / threadsPerBlock;

    generate_automata_chaotic<<<numKerBlocksChaotic, threadsPerBlock>>>(
        d_automatas, d_chaotic_values, num_blocks, d_indices,
        block_length);
        
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_automatas);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("generate_automata_chaotic execution failed");
    }
    cudaDeviceSynchronize();
    
    const int numKerBlocksSort = (num_blocks + threadsPerBlock - 1) / threadsPerBlock;

    sort_indices_by_chaotic_values_global<unsigned short><<<numKerBlocksSort, threadsPerBlock>>>(
        d_chaotic_values,num_blocks,d_indices,block_length);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_automatas);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("sort_indices_by_chaotic_values_global synchronization failed");
    }
    
    cudaFree(d_automatas);
    cudaFree(d_chaotic_values);
    
    return d_indices;
}

__host__ void block_phase_permutation(
    unsigned char* d_image, unsigned char* d_image_out, unsigned int *block_permutations, size_t cols, size_t rows, size_t block_size)
{
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    permute_blocks_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image, d_image_out, block_permutations, block_size, cols, rows);

    cudaDeviceSynchronize();


}

__host__ void rows_and_columns_permutation(
    unsigned char* d_image, unsigned char* d_image_out, unsigned int *d_row_permutations, unsigned int *d_col_permutations, size_t cols, size_t rows,bool inverse)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (!inverse) {
        permute_rows_kernel<<<numBlocks, threadsPerBlock>>>(
            d_image, d_image_out, d_row_permutations, cols, rows);

        if (cudaGetLastError() != cudaSuccess) {
            throw std::runtime_error("Row permutation error");
        }

        cudaDeviceSynchronize();
        
        permute_columns_kernel<<<numBlocks, threadsPerBlock>>>(
            d_image_out, d_image, d_col_permutations, cols, rows);
        if (cudaGetLastError() != cudaSuccess) {
            throw std::runtime_error("Col permutation error");
        }
    } else{
        permute_columns_kernel<<<numBlocks, threadsPerBlock>>>(
            d_image, d_image_out, d_col_permutations, cols, rows);
        if (cudaGetLastError() != cudaSuccess) {
            throw std::runtime_error("Col permutation error");
        }
            cudaDeviceSynchronize();
            
        permute_rows_kernel<<<numBlocks, threadsPerBlock>>>(
            d_image_out, d_image, d_row_permutations, cols, rows);
        if (cudaGetLastError() != cudaSuccess) {
            throw std::runtime_error("Row permutation error");
        }
    }

    cudaDeviceSynchronize();
}

__host__ void flow_encrypt(
    unsigned char *image,
    unsigned char *image_out,
    const std::vector<unsigned char> seeds,
    size_t cols,
    size_t rows,
    double r,
    int rounds){

    unsigned char* d_seeds = nullptr;

    cudaMalloc(&d_seeds,seeds.size() * sizeof(unsigned char));
    cudaMemcpy(d_seeds,seeds.data(),seeds.size() * sizeof(unsigned char),cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess) {
            cudaFree(d_seeds);
            throw std::runtime_error("Seeds copy error.");
        }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    flow_encrypt_recursive<<<numBlocks, threadsPerBlock>>>(image, image_out, seeds.data(), cols, rows, r, rounds);
    if (cudaGetLastError() != cudaSuccess) {
            cudaFree(d_seeds);
            throw std::runtime_error("Flow encryption error");
        }
    cudaFree(d_seeds);
}