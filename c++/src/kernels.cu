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
                                   const unsigned char *seeds,
                                   Image_dimnesions img_dimensions, double r,
                                   int rounds) {
  // Each thread processes one column; uses `uno` to generate XOR mask values.
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= img_dimensions.cols)
    return;

  double xn = seeds[x] / 255.0;

  for (int y = 0; y < img_dimensions.rows; y++) {
    xn = uno(xn, r);

    int idx = y * img_dimensions.cols + x;

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

__global__ void keystream_generation(D_pointers d_pointers,
                                     Image_dimnesions img_dimensions,
                                     double r) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= img_dimensions.cols)
    return;

  double xn = d_pointers.d_seeds[x] / 255.0;
  unsigned char mixed = 0;

  int idx = x;

  int stride = img_dimensions.cols;

  for (int k = 0; k < 200; k++) {
    xn = uno(xn, r);
  }

  for (int y = 0; y < img_dimensions.rows; y++) {
    xn = uno(xn, r);

    unsigned long long u = __double_as_longlong(xn);

    unsigned char b1 = (u >> 4) & 0xFF;
    unsigned char b2 = (u >> 12) & 0xFF;

    mixed = (b1 ^ ((b2 << 3) | (b2 >> 5))) + (b1 >> 2);

    d_pointers.d_flow[idx] = mixed;

    idx += stride;
  }
  d_pointers.d_seeds[x] = mixed;
}

__global__ void permute_blocks_kernel(unsigned char *image,
                                      unsigned char *image_out,
                                      unsigned int *permutations,
                                      size_t block_size,
                                      Image_dimnesions img_dimensions) {
  // 1. Global thread coordinates (Destination Pixel)
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // 2. Destination Boundary Check
  // Ensure we don't write outside the output buffer.
  if (x >= img_dimensions.cols || y >= img_dimensions.rows)
    return;

  // 3. Identify Macro-Block coordinates
  int block_idx_x = x / block_size;
  int block_idx_y = y / block_size;

  // 4. Identify Local coordinates within the block
  int local_x = x % block_size;
  int local_y = y % block_size;

  // 5. Calculate the linear ID of the current block
  int blocks_per_row = img_dimensions.cols / block_size;
  int current_block_linear_id = block_idx_y * blocks_per_row + block_idx_x;

  // 6. Calculate pointer offset for this specific block
  // Jump to the start of the permutation table for this specific block ID.
  size_t block_data_offset =
      (size_t)current_block_linear_id * (block_size * block_size);

  // 7. Lookup the permutation
  // Retrieve the 1D index where the source pixel should come from.
  unsigned int src_permuted_index =
      permutations[block_data_offset + (local_y * block_size + local_x)];

  // 8. Calculate Source Global Coordinates
  // Convert the permuted 1D index back to 2D global coordinates.
  int src_local_x = src_permuted_index % block_size;
  int src_local_y = src_permuted_index / block_size;

  int src_global_x = block_idx_x * block_size + src_local_x;
  int src_global_y = block_idx_y * block_size + src_local_y;

  // 9. Source Boundary Check
  if (src_global_x < img_dimensions.cols &&
      src_global_y < img_dimensions.rows) {
    image_out[y * img_dimensions.cols + x] =
        image[src_global_y * img_dimensions.cols + src_global_x];
  }
}

__global__ void permute_blocks_kernel_simple(unsigned char *image,
                                             unsigned char *image_out,
                                             unsigned int *permutation,
                                             unsigned int *permutation_inverse,
                                             size_t block_size,
                                             Image_dimnesions img_dimensions) {
  // Calculate global thread coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Boundary check
  if (x >= img_dimensions.cols || y >= img_dimensions.rows)
    return;

  // 1. Identify the block coordinates (Macro-coordinates)
  int block_idx_x = x / block_size;
  int block_idx_y = y / block_size;

  // 2. Calculate local position within the block (0 to block_size-1)
  int local_x = x % block_size;
  int local_y = y % block_size;

  // 3. Select permutation array based on Block Parity (Chessboard pattern)
  // Using block coordinates for parity avoids high-frequency noise artifacts
  // compared to using pixel coordinates.
  unsigned int *current_permutation =
      ((block_idx_x + block_idx_y) & 1) ? permutation : permutation_inverse;

  // 4. Retrieve the source local index from the selected permutation table
  int permuted_index = current_permutation[local_y * block_size + local_x];

  // Decode 1D index back to 2D local source coordinates
  int src_local_x = permuted_index % block_size;
  int src_local_y = permuted_index / block_size;

  // 5. Calculate the Global Source Coordinates
  // (Block Origin + Permuted Local Offset)
  int src_global_x = block_idx_x * block_size + src_local_x;
  int src_global_y = block_idx_y * block_size + src_local_y;

  // Perform the copy if source is within bounds
  if (src_global_x < img_dimensions.cols &&
      src_global_y < img_dimensions.rows) {
    // Scattered Read (Random access from source) -> Coalesced Write (Linear
    // access to dest)
    image_out[y * img_dimensions.cols + x] =
        image[src_global_y * img_dimensions.cols + src_global_x];
  }
}

__global__ void permute_rows_kernel(unsigned char *image,
                                    unsigned char *image_out,
                                    unsigned int *permutation,
                                    Image_dimnesions img_dimensions) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < img_dimensions.cols && y < img_dimensions.rows) {
    image_out[y * img_dimensions.cols + x] =
        image[permutation[y] * img_dimensions.cols + x];
  }
}

__global__ void permute_columns_kernel(unsigned char *image,
                                       unsigned char *image_out,
                                       unsigned int *permutation,
                                       Image_dimnesions img_dimensions) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < img_dimensions.cols && y < img_dimensions.rows) {
    image_out[y * img_dimensions.cols + x] =
        image[y * img_dimensions.cols + permutation[x]];
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