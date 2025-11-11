/**
 * @file kernels.cu
 * @brief CUDA kernels used by the encryption pipeline: flow generator and
 * permutation kernels.
 */

#include "../include/kernels.cuh"

// Device helpers and kernels for the flow generator and permutations.
__device__ double uno(double x, double r) {
  double t = r + 3.0 * x * x;
  return fabs(cos(3.14159265 * r * cos(3.14159265 * t) * t));
}

__global__ void keystream_to_image(unsigned char *image,
                                   unsigned char *image_out,
                                   const unsigned char *seeds, int width,
                                   int height, double r, int rounds) {
  // Each thread processes one column; uses `uno` to generate XOR mask values.
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= width)
    return;

  double xn = seeds[x] / 255.0;

  for (int y = 0; y < height; y++) {
    xn = uno(xn, r);

    int idx = y * width + x;

    union {
      double f;
      unsigned long long u;
    } conv;
    conv.f = xn;

    unsigned char b1 = (conv.u >> 4) & 0xFF;
    unsigned char b2 = (conv.u >> 12) & 0xFF;
    unsigned char mixed = (b1 ^ ((b2 << 3) | (b2 >> 5))) + (b1 >> 2);

    image_out[idx] = image[idx] ^ mixed;
  }
}

__global__ void keystream_generation(unsigned char *keystream_out,
                                     const unsigned char *seeds, int width,
                                     int height, double r, int rounds) {
  // Each thread processes one column; uses `uno` to generate XOR mask values.
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= width)
    return;

  double xn = seeds[x] / 255.0;

  for (int y = 0; y < height; y++) {
    xn = uno(xn, r);

    int idx = y * width + x;

    union {
      double f;
      unsigned long long u;
    } conv;
    conv.f = xn;

    unsigned char b1 = (conv.u >> 4) & 0xFF;
    unsigned char b2 = (conv.u >> 12) & 0xFF;
    unsigned char mixed = (b1 ^ ((b2 << 3) | (b2 >> 5))) + (b1 >> 2);

    keystream_out[idx] = mixed;
  }
}

__global__ void permute_blocks_kernel(unsigned char *image,
                                      unsigned char *image_out,
                                      unsigned int *permutations,
                                      size_t block_size, size_t cols,
                                      size_t rows) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int number_block_per_row = cols / block_size;

  if (x < cols && y < rows) {

    // Compute block number
    int block = y / block_size * (cols / block_size) + x / block_size;

    // Position inside the block
    int block_y = y % block_size;
    int block_x = x % block_size;

    // Index inside the flattened permutation for this block
    int src_permuted_index = permutations[block * block_size * block_size +
                                          block_y * block_size + block_x];

    // Now compute the coordinates inside the block of the source pixel
    block_x = src_permuted_index % block_size;
    block_y = src_permuted_index / block_size;

    int pixel_y = block / number_block_per_row * block_size + block_y;
    int pixel_x = block % number_block_per_row * block_size + block_x;

    image_out[y * cols + x] = image[pixel_y * cols + pixel_x];
  }
}

__global__ void permute_rows_kernel(unsigned char *image,
                                    unsigned char *image_out,
                                    unsigned int *permutation, size_t cols,
                                    size_t rows) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < cols && y < rows) {
    image_out[y * cols + x] = image[permutation[y] * cols + x];
  }
}

__global__ void permute_columns_kernel(unsigned char *image,
                                       unsigned char *image_out,
                                       unsigned int *permutation, size_t cols,
                                       size_t rows) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < cols && y < rows) {
    image_out[y * cols + x] = image[y * cols + permutation[x]];
  }
}

__global__ void generate_chaotic(unsigned char *passwords, size_t num_blocks,
                                 double *chaotic_vals, unsigned int *indices,
                                 double r, size_t block_length,
                                 size_t transition_length) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_blocks)
    return;

  double x =
      (static_cast<double>(passwords[idx]) + 1.0) / 257.0; // Normalize to (0,1)

  for (int i = 0; i < transition_length; ++i) {
    x = uno(x, r);
  }

  int base_idx = idx * block_length;
  for (int i = 0; i < block_length; i++) {
    x = uno(x, r);
    chaotic_vals[base_idx + i] = x;
    indices[base_idx + i] = i;
  }

  sort_indices_by_chaotic_values<double>(base_idx, chaotic_vals, indices,
                                         block_length);
}

__global__ void generate_automata_chaotic(unsigned int **automata_states,
                                          unsigned short *d_chaotic_values,
                                          size_t num_blocks,
                                          unsigned int *indices,
                                          size_t block_length) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_blocks * block_length)
    return;
  unsigned int *automata_state = automata_states[idx / block_length];
  if (idx & 1)
    d_chaotic_values[idx] = automata_state[idx] >> 16;
  else
    d_chaotic_values[idx] = automata_state[idx] & 0x0000FFFF;
  indices[idx] = idx;
}