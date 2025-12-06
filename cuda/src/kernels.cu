/**
 * @file kernels.cu
 * @brief CUDA kernels used by the encryption pipeline: flow generator and
 * permutation kernels.
 */

#include "../include/kernels.cuh"
#include <climits>

__global__ void permute_blocks_kernel(unsigned char *image,
                                      unsigned char *image_out,
                                      unsigned int *permutations,
                                      size_t block_size,
                                      Image_dimensions img_dimensions) {
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
                                             Image_dimensions img_dimensions) {
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
                                    Image_dimensions img_dimensions) {
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
                                       Image_dimensions img_dimensions) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < img_dimensions.cols && y < img_dimensions.rows) {
    image_out[y * img_dimensions.cols + x] =
        image[y * img_dimensions.cols + permutation[x]];
  }
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

__global__ void deinterleave_channels_kernel(const unsigned char *input,
                                             unsigned char *output,
                                             int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  // Input pointer treats image as 3-channel interleaved (width * 3 bytes per row)
  // But strictly speaking, input is uchar array.
  // OpenCV stores BGR: Pixel at (x,y) is input[(y*width + x)*3 + c]
  
  int input_idx_base = (y * width + x) * 3;
  
  unsigned char b = input[input_idx_base + 0];
  unsigned char g = input[input_idx_base + 1];
  unsigned char r = input[input_idx_base + 2];

  // Output logic: Side-by-side [B][G][R]
  // Total width of output is 3*width.
  // B is at (x, y)
  // G is at (x + width, y)
  // R is at (x + 2*width, y)
  
  // Stride of output is 3*width. 
  int out_stride = width * 3;
  
  output[y * out_stride + x]             = b;
  output[y * out_stride + (x + width)]   = g;
  output[y * out_stride + (x + 2*width)] = r;
}

__global__ void interleave_channels_kernel(const unsigned char *input,
                                           unsigned char *output,
                                           int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // We iterate over the ORIGINAL image dimensions (WxH)
  if (x >= width || y >= height) return;
  
  // Input maps:
  // B: (x, y)
  // G: (x + width, y)
  // R: (x + 2*width, y)
  // Input stride is 3*width (concatenated image width)
  
  int in_stride = width * 3;
  
  unsigned char b = input[y * in_stride + x];
  unsigned char g = input[y * in_stride + (x + width)];
  unsigned char r = input[y * in_stride + (x + 2*width)];
  
  // Output: Interleaved BGR.
  // Stride: width * 3
  
  int output_idx_base = (y * width + x) * 3;
  
  output[output_idx_base + 0] = b;
  output[output_idx_base + 1] = g;
  output[output_idx_base + 2] = r;
}
