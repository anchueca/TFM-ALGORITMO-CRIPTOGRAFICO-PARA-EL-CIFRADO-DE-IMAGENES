/**
 * @file kernels.cu
 * @brief CUDA kernels used by the encryption pipeline: flow generator and
 * permutation kernels.
 */

#include "../include/kernels.cuh"
#include <climits>

__device__ __forceinline__ Real coupled_map(Real c_seed, Real r_seed,
                                            Real l_seed, Real r,
                                            unsigned short *celular_automata) {

  Real r_next = chaotic_functio(r_seed, r);
  Real c_next = chaotic_functio(c_seed, r);
  Real l_next = chaotic_functio(l_seed, r);

  evolve_16bit_isolated(celular_automata, 30, 1); // Evolution of the 16-bit CA

  // Extract weights from the CA state
  unsigned short ca_val = *celular_automata;
  Real v1 = static_cast<Real>((ca_val >> 8) & 0xFF) /
            255.0; // First 8 bits normalized (0, 1)
  Real v2 =
      static_cast<Real>(ca_val & 0xFF) / 255.0; // Last 8 bits normalized (0, 1)

  // Distribution of influence:
  // v1 determines the proportion of c_next.
  // v2 determines the proportion of r_next and l_next of the rest (1 - v1).
  Real c_influence = v1;
  Real rest = (Real)1.0 - v1;
  Real r_influence = rest * v2;
  Real l_influence = rest * ((Real)1.0 - v2);

  return (c_next * c_influence) + (r_next * r_influence) +
         (l_next * l_influence);
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
                                             unsigned char *output, int width,
                                             int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Input pointer treats image as 3-channel interleaved (width * 3 bytes per
  // row) But strictly speaking, input is uchar array. OpenCV stores BGR: Pixel
  // at (x,y) is input[(y*width + x)*3 + c]

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

  output[y * out_stride + x] = b;
  output[y * out_stride + (x + width)] = g;
  output[y * out_stride + (x + 2 * width)] = r;
}

__global__ void interleave_channels_kernel(const unsigned char *input,
                                           unsigned char *output, int width,
                                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // We iterate over the ORIGINAL image dimensions (WxH)
  if (x >= width || y >= height)
    return;

  // Input maps:
  // B: (x, y)
  // G: (x + width, y)
  // R: (x + 2*width, y)
  // Input stride is 3*width (concatenated image width)

  int in_stride = width * 3;

  unsigned char b = input[y * in_stride + x];
  unsigned char g = input[y * in_stride + (x + width)];
  unsigned char r = input[y * in_stride + (x + 2 * width)];

  // Output: Interleaved BGR.
  // Stride: width * 3

  int output_idx_base = (y * width + x) * 3;

  output[output_idx_base + 0] = b;
  output[output_idx_base + 1] = g;
  output[output_idx_base + 2] = r;
}

__global__ void keystream_generation_parallel(
    unsigned char *__restrict__ d_flow, Real *__restrict__ d_seeds,
    unsigned short *celular_automata, unsigned short *image_automata_state,
    Image_dimensions img_dimensions, Real r, const size_t total_steps,
    Real *__restrict__ d_chaotic_values_for_permutation,
    size_t permutation_block_size, size_t transition_length, size_t numBlocks) {

  // Shared size is determined at kernel launch
  extern __shared__ char shared_mem[];
  Real *s_seeds = reinterpret_cast<Real *>(shared_mem);

  // Thread ID
  const int tid = threadIdx.x;
  const int x = blockIdx.x * blockDim.x + tid;

  const size_t cols_and_blocks = img_dimensions.cols + numBlocks; // Extra seed
  if (x >= cols_and_blocks)
    return;

  // Get neighbors

  Real *c_seed = nullptr;
  Real *r_seed = nullptr;
  Real *l_seed = nullptr;
  {
    size_t block_length = blockIdx.x < numBlocks
                              ? blockDim.x
                              : cols_and_blocks - (numBlocks - 1) * blockDim.x;

    c_seed = &s_seeds[tid];
    r_seed = &s_seeds[tid == 0 ? block_length - 1 : tid - 1];
    l_seed = &s_seeds[tid == block_length - 1 ? 0 : tid + 1];
  }

  // Get initial state
  Real current_xn = 0.0;
  unsigned short celular_automata_value;

  size_t state_idx;
  if (tid == 0) {
    state_idx = img_dimensions.cols +
                blockIdx.x; // Unique index for extra seed to avoid race
    current_xn = d_seeds[state_idx];
    // If d_chaotic_values_for_permutation is not null, it's the first call
    // (transition/permutation). We use image_automata_state[0] (the initialized
    // hash) as the collective seed.
    celular_automata_value = d_chaotic_values_for_permutation != nullptr
                                 ? image_automata_state[0]
                                 : image_automata_state[blockIdx.x];
  } else {
    state_idx = x - (blockIdx.x + 1);
    current_xn = d_seeds[state_idx];
    celular_automata_value = celular_automata[state_idx];
  }
  *c_seed = current_xn;

  for (size_t step = 0; step < total_steps; ++step) {
    __syncthreads(); // To avoid race conditions
    current_xn =
        coupled_map(current_xn, *r_seed, *l_seed, r, &celular_automata_value);
    // Race condition fix: Only one block (Block 0) should write to
    // d_chaotic_values_for_permutation
    if (blockIdx.x == 0 && tid == 0 &&
        d_chaotic_values_for_permutation != nullptr &&
        step >= total_steps - permutation_block_size) {
      *c_seed = current_xn;
      d_chaotic_values_for_permutation[step -
                                       (total_steps - permutation_block_size)] =
          current_xn;
    } else if (tid != 0 && step >= transition_length) {
      size_t row = step - transition_length;
      d_flow[row * img_dimensions.cols + state_idx] =
          convertToBitStream(current_xn);
    }

    __syncthreads();
    *c_seed = current_xn;
  }

  // Store final state back to global memory (Include tid=0 to persist extra
  // seeds)
  d_seeds[state_idx] = current_xn;
  // For normal threads, persist CA. For tid=0, we use a constant so no need to
  // save back
  if (tid != 0) {
    celular_automata[state_idx] = celular_automata_value;
  } else {
    image_automata_state[blockIdx.x] =
        celular_automata_value ^ convertToBitStream(current_xn);
  }
}

__global__ void convert_bits_to_real_kernel(Real *d_seeds,
                                            size_t num_elements) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= num_elements)
    return;
  d_seeds[idx] =
      static_cast<Real>(reinterpret_cast<uint32_t *>(d_seeds)[idx]) / UINT_MAX;
}

__device__ void sort_indices_by_chaotic_values(int base_idx, Real *chaotic_vals,
                                               unsigned int *indices,
                                               size_t block_length) {

  Real local_vals[MAX_BLOCK_SIZE];
  unsigned int local_indices[MAX_BLOCK_SIZE];

  if (block_length > MAX_BLOCK_SIZE)
    printf("Length too large");

  for (size_t i = 0; i < block_length; i++) {
    local_vals[i] = chaotic_vals[base_idx + i];
    local_indices[i] = indices[base_idx + i];
  }

  for (size_t i = 1; i < block_length; i++) {
    Real key_val = local_vals[i];
    unsigned int key_idx = local_indices[i];

    int j = (int)i - 1;

    while (j >= 0 && local_vals[j] > key_val) {
      local_vals[j + 1] = local_vals[j];
      local_indices[j + 1] = local_indices[j];
      j = j - 1;
    }

    local_vals[j + 1] = key_val;
    local_indices[j + 1] = key_idx;
  }

  for (size_t i = 0; i < block_length; i++) {
    indices[base_idx + i] = local_indices[i];
    chaotic_vals[base_idx + i] = local_vals[i];
  }
}

__global__ void global_seed_mix_kernel(Real *d_seeds, size_t offset,
                                       size_t n_blocks) {
  if (n_blocks == 0)
    return;

  Real sum = 0;
  // Step 1: Accumulate all extra seeds
  for (size_t i = 0; i < n_blocks; i++) {
    sum += d_seeds[offset + i];
  }

  // Step 2: Calculate mean
  Real mean = sum / (Real)n_blocks;

  // Step 3: Apply iterative coupling: (S + Mean) / 2
  for (size_t i = 0; i < n_blocks; i++) {
    d_seeds[offset + i] = (d_seeds[offset + i] + mean) / 2.0;
  }
}