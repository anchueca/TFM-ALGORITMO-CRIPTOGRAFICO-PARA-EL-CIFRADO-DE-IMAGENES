#include "../include/kernels.cuh"
#include <cstddef>

__device__ double uno(double x, double r)
{
    double t = r + 3.0 * x * x;
    return fabs(cos(3.14159265 * r * cos(3.14159265 * t) * t));
}
/*
__global__ void flow_encrypt_recursive(
    unsigned char *image,
    float *seeds,
    int width,
    int height,
    float r,
    int rounds)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_start = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y_start >= height)
        return;

    float xn = seeds[y_start];

    for (int r_idx = 0; r_idx < rounds; ++r_idx)
    {
        int y = (y_start + r_idx) % height;
        int idx = y * width + x;

        for (int i = 0; i <= x; ++i)
        {
            xn = uno(xn, r);
        }

        union
        {
            float f;
            unsigned int u;
        } conv;
        conv.f = xn;

        unsigned int mantisa = conv.u & 0x007FFFFF;
        unsigned char b1 = (mantisa >> 4) & 0xFF;
        unsigned char b2 = (mantisa >> 12) & 0xFF;
        unsigned char mixed = (b1 ^ ((b2 << 3) | (b2 >> 5))) + (b1 >> 2);

        image[idx] ^= mixed;
    }
}
*/
__global__ void permute_blocks_kernel(
    unsigned char *image, int *permutations,
    size_t block_size, size_t blocks_per_row)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=0; i<block_size; i++) {
        unsigned char temp = image[idx * block_size + i];
        image[idx * block_size + i] =
            image[idx * block_size + permutations[idx * block_size + i]];
        image[idx * block_size + permutations[idx * block_size + i]] = temp;
    }
}

/*__global__ void invert_permutations_kernel(int *permutations, int *inverses, int length)
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
}*/

/// @brief Generate a set of permutations based on chaotic sequences derived from block passwords.
/// @param block_passwords Vector de contraseñas de los bloques.
/// @param block_length La longitud de los bloques.
/// @param num_blocks El número de bloques.
/// @return Un vector de vectores de enteros que representa las permutaciones generadas.
__host__ std::vector<std::vector<int>> generate_permutations(const std::vector<unsigned char> block_passwords, size_t block_length, size_t num_blocks)
{
    if (block_passwords.size() < num_blocks) {
        throw std::runtime_error("Insufficient passwords for blocks");
    }

    std::vector<std::vector<int>> indices(num_blocks, std::vector<int>(block_length));
    
    size_t total_size = num_blocks * block_length;
    
    unsigned char *d_passwords = nullptr;
    int *d_indices = nullptr;
    double *d_chaotic_values = nullptr;
    
    cudaError_t err = cudaMalloc(&d_passwords, num_blocks * sizeof(unsigned char));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for passwords");
    }
    
    err = cudaMalloc(&d_indices, total_size * sizeof(int));
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
        cudaFree(d_indices);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("Kernel execution failed");
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_indices);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("Kernel synchronization failed");
    }
    
    // return values
    std::vector<int> flat_indices(total_size);
    err = cudaMemcpy(flat_indices.data(), d_indices, total_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_indices);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("Failed to copy results from device");
    }
    
    cudaFree(d_passwords);
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);
    
    for (int i = 0; i < num_blocks; ++i) {
        std::copy(
            flat_indices.begin() + i * block_length,
            flat_indices.begin() + (i + 1) * block_length,
            indices[i].begin()
        );
    }
    
    return indices;
}

__global__ void generate_chaotic(unsigned char* passwords, size_t num_blocks, double* chaotic_vals, int *indices, double r, size_t block_length, size_t transition_length)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_blocks) return;

    double x = (static_cast<double>(passwords[idx]) + 1.0) / 257.0;  // Normalizar a (0,1)

    for (int i = 0; i < transition_length; ++i) {
        x = uno(x, r);
    }

    int base_idx = idx * block_length;
    for (int i = 0; i < block_length; i++) {
        x = uno(x, r);
        chaotic_vals[base_idx + i] = x;
        indices[base_idx + i] = i;
    }

    __syncthreads();

    for (int i = 0; i < block_length - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < block_length; j++) {
            if (chaotic_vals[base_idx + j] < chaotic_vals[base_idx + min_idx]) {
                min_idx = j;
            }
        }

        if (min_idx != i) {
            double temp_val = chaotic_vals[base_idx + i];
            chaotic_vals[base_idx + i] = chaotic_vals[base_idx + min_idx];
            chaotic_vals[base_idx + min_idx] = temp_val;

            int temp_idx = indices[base_idx + i];
            indices[base_idx + i] = indices[base_idx + min_idx];
            indices[base_idx + min_idx] = temp_idx;
        }
    }
}

__host__ void block_phase_permutation(
    cv::cuda::GpuMat image, std::vector<std::vector<int>> &block_permutations)
{
    unsigned char *d_image = image.ptr<unsigned char>();
    int *d_permutations = nullptr;
    
    size_t block_number = block_permutations.size();
    size_t block_size = block_permutations[0].size();
    size_t image_size = image.rows * image.cols * sizeof(unsigned char);
    size_t permutations_size = block_size * block_number * sizeof(int);
    size_t blocks_per_row = image.cols / block_size;

    cudaMalloc(&d_permutations, permutations_size);

    cudaMemcpy(d_permutations, block_permutations.data(), permutations_size, cudaMemcpyHostToDevice);

    permute_blocks_kernel<<<32, 128>>>(
        d_image, d_permutations, block_size, blocks_per_row);

    cudaDeviceSynchronize();

    cudaMemcpy(image.data, d_image, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_permutations);

}
