#include "../include/kernels.cuh"

__device__ double uno(double x, double r)
{
    double t = r + 3.0 * x * x;
    return fabs(cos(3.14159265 * r * cos(3.14159265 * t) * t));
}

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

__global__ void permute_blocks_kernel(
    unsigned char *input,
    unsigned char *output,
    int *permutations,
    int block_height,
    int block_width,
    int image_rows,
    int image_cols,
    int num_blocks_row,
    int num_blocks_col,
    int channels)
{
    int block_size = block_height * block_width;
    int total_blocks = num_blocks_row * num_blocks_col;
    int total_threads = block_size * total_blocks;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= total_threads)
        return;

    int block_id = tid / block_size;
    int pixel_id = tid % block_size;

    int block_row = block_id / num_blocks_col;
    int block_col = block_id % num_blocks_col;

    int perm_index = block_id * block_size + pixel_id;
    int src_pos = permutations[perm_index];

    int dst_y = block_row * block_height + pixel_id / block_width;
    int dst_x = block_col * block_width + pixel_id % block_width;

    int src_y = block_row * block_height + src_pos / block_width;
    int src_x = block_col * block_width + src_pos % block_width;

    for (int c = 0; c < channels; ++c)
    {
        if (dst_y < image_rows && dst_x < image_cols && src_y < image_rows && src_x < image_cols)
        {
            output[(dst_y * image_cols + dst_x) * channels + c] =
                input[(src_y * image_cols + src_x) * channels + c];
        }
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

/// @brief Genera una permutación a partir de la contraseña del bloque.
/// @param block_passwords Vector de contraseñas de los bloques.
/// @param block_length La longitud de los bloques.
/// @param num_blocks El número de bloques.
/// @return Un vector de vectores de enteros que representa las permutaciones generadas.
__host__ std::vector<std::vector<int>> generate_permutations(const std::vector<unsigned char> block_passwords, int block_length, int num_blocks)
{
    unsigned char *d_passwords;
    int *d_indices;
    double *d_chaotic_values;

    int chaos_memory_length = num_blocks * block_length;

    cudaMalloc(&d_passwords, num_blocks * sizeof(unsigned char));
    cudaMalloc(&d_indices, chaos_memory_length * sizeof(int));
    cudaMalloc(&d_chaotic_values, chaos_memory_length * sizeof(double));

    cudaMemcpy(d_passwords, block_passwords.data(), num_blocks * sizeof(unsigned char), cudaMemcpyHostToDevice);

    double r = 0.998; // Control parameter for the chaotic map
    int transition_length = 20;

    int threads = 256;
    int blocks  = (num_blocks + threads - 1) / threads;
    generate_chaotic<<<blocks, threads>>>(d_passwords, num_blocks, d_chaotic_values, d_indices, r, block_length, transition_length);

    cudaDeviceSynchronize();

    std::vector<std::vector<int>> indices(num_blocks, std::vector<int>(block_length));
    cudaMemcpy(indices[0].data(), d_indices, chaos_memory_length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_passwords);
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);

    return indices;
}

__global__ void generate_chaotic(unsigned char* passwords, int num_blocks, double* chaotic_vals, int *indices, double r, int block_length, int transition_length)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= gridDim.x * blockDim.x) return;
    if (idx >= num_blocks) return;


    double x = static_cast<double>(passwords[idx]);

    for (int i = 0; i < transition_length; ++i)
        x = uno(x, r);

    for (int i = 0; i < block_length; i++)
    {
        chaotic_vals[idx * block_length + i] = 5;
        indices[idx * block_length + i] = i;
        x = uno(x, r);
    }

    // sort_indices_by_chaotic_values(chaotic_vals + idx * length, indices + idx * length, length);
}

__host__ void block_phase_permutation(
    cv::Mat &image, int num_rows, int num_cols, int block_height, int block_width, const std::vector<int> &block_permutations)
{
    unsigned char *d_input;
    unsigned char *d_output;
    int *d_permutations;

    int channels = image.channels();
    int image_size = num_rows * num_cols * channels * sizeof(unsigned char);
    int block_size = block_height * block_width;
    int total_blocks = (num_rows / block_height) * (num_cols / block_width);
    int permutations_size = total_blocks * block_size * sizeof(int);

    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);
    cudaMalloc(&d_permutations, permutations_size);

    cudaMemcpy(d_input, image.data, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_permutations, block_permutations.data(), permutations_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((num_cols + blockDim.x - 1) / blockDim.x, (num_rows + blockDim.y - 1) / blockDim.y);

    permute_blocks_kernel<<<gridDim, blockDim>>>(
        d_input, d_output, d_permutations,
        block_height, block_width,
        num_rows, num_cols,
        num_rows / block_height, num_cols / block_width,
        channels);

    cudaDeviceSynchronize();

    cudaMemcpy(image.data, d_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_permutations);
}
