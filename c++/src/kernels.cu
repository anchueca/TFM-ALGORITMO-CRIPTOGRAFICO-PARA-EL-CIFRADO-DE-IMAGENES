#include "../include/kernels.cuh"

__device__ double uno(double x, double r)
{
    double t = r + 3.0 * x * x;
    return fabs(cos(3.14159265 * r * cos(3.14159265 * t) * t));
}

__global__ void flow_encrypt_recursive(
    unsigned char *image,
    unsigned char *image_out,
    const unsigned char *seeds,
    int width,
    int height,
    double r,
    int rounds)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_start = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y_start >= height)
        return;

    double xn = seeds[y_start] / static_cast<double>(INT_MAX);

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
            double f;
            unsigned int u;
        } conv;
        conv.f = xn;

        unsigned int mantisa = conv.u & 0x007FFFFF;
        unsigned char b1 = (mantisa >> 4) & 0xFF;
        unsigned char b2 = (mantisa >> 12) & 0xFF;
        unsigned char mixed = (b1 ^ ((b2 << 3) | (b2 >> 5))) + (b1 >> 2);

        image_out[idx] = image[idx] ^ mixed;
    }
}


//block_size: length of one side of a block
__global__ void permute_blocks_kernel(
    unsigned char *image, unsigned char *image_out, unsigned int *permutations,
    size_t block_size, size_t cols, size_t rows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int number_block_per_row = cols / block_size;

    if (x < cols && y < rows) {

        // The block number is
        int block = y / block_size * (cols / block_size) + x / block_size;

        // Position inside the block
        int block_y = y % block_size;
        int block_x = x % block_size;

        // Index to permutate
        int src_permuted_index = permutations[block * block_size * block_size + block_y * block_size + block_x];

        // Now are the coordinates inside the block of the source pixel
        block_x= src_permuted_index % block_size;
        block_y= src_permuted_index / block_size;

        int pixel_y = block / number_block_per_row * block_size + block_y;
        int pixel_x = block % number_block_per_row * block_size + block_x;

        image_out[y * cols + x] = image[pixel_y * cols + pixel_x];

    }
}

__global__ void permute_rows_kernel(
    unsigned char *image, unsigned char *image_out, unsigned int *permutation, size_t cols, size_t rows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        image_out[y * cols + x] = image[permutation[y] * cols + x];
    }
}

__global__ void permute_columns_kernel(
    unsigned char *image, unsigned char *image_out, unsigned int *permutation, size_t cols, size_t rows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        image_out[y * cols + x] = image[y * cols + permutation[x]];
    }
}

__global__ void generate_chaotic(unsigned char* passwords, size_t num_blocks, double* chaotic_vals, unsigned int *indices, double r, size_t block_length, size_t transition_length)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_blocks) return;

    double x = (static_cast<double>(passwords[idx]) + 1.0) / 257.0;  // Normalize to (0,1)

    for (int i = 0; i < transition_length; ++i) {
        x = uno(x, r);
    }

    int base_idx = idx * block_length;
    for (int i = 0; i < block_length; i++) {
        x = uno(x, r);
        chaotic_vals[base_idx + i] = x;
        indices[base_idx + i] = i;
    }

    sort_indices_by_chaotic_values<double>(base_idx,chaotic_vals, indices, block_length);
}

__global__ void generate_automata_chaotic(unsigned int** automata_states, unsigned int* d_chaotic_values, size_t num_blocks, unsigned int *indices, size_t block_length)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_blocks) return;
    int base_idx = idx * block_length;
    //TODO: Tengo que tomar los punteros en automata_states y copiar su contenido en chaotic_values
    unsigned int* automata_state = automata_states[idx];
    for(int i=0;i<block_length;i++ ){
        d_chaotic_values[base_idx+i] = automata_state[i];
    }

    sort_indices_by_chaotic_values<unsigned int>(base_idx,d_chaotic_values, indices, block_length);
}
