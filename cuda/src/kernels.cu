/**
 * @file kernels.cu
 * @brief CUDA kernels used by the encryption pipeline: flow generator and
 * permutation kernels.
 */

#include "../include/kernels.cuh"
#include <climits>

__device__ __forceinline__ Real coupled_map(Real c_next, Real r_next,
                                            Real l_next,
                                            unsigned short *cellular_automata) {

  evolve_16bit_isolated(cellular_automata, 30, 1); // Evolution of the 16-bit CA

  // Extract weights from the CA state
  unsigned short ca_val = *cellular_automata;
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

__global__ void permute_blocks_kernel_simple(const unsigned char *__restrict__ image,
                                              unsigned char *__restrict__ image_out,
                                              const unsigned int *__restrict__ permutation,
                                              const unsigned int *__restrict__ permutation_inverse,
                                             size_t block_size,
                                             Image_dimensions img_dimensions) {
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    int x = block_x * block_size + local_x;
    int y = block_y * block_size + local_y;

    if (x >= img_dimensions.cols || y >= img_dimensions.rows)
        return;

    bool use_inverse = (block_x + block_y) & 1;
    const unsigned int* perm =
        use_inverse ? permutation_inverse : permutation;

    int permuted_index = perm[local_y * block_size + local_x];

    int src_local_x = permuted_index % block_size;
    int src_local_y = permuted_index / block_size;

    int src_x = block_x * block_size + src_local_x;
    int src_y = block_y * block_size + src_local_y;

    if (src_x < img_dimensions.cols && src_y < img_dimensions.rows) {
        image_out[y * img_dimensions.cols + x] =
            image[src_y * img_dimensions.cols + src_x];
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

__global__ void generate_automata_chaotic(const unsigned int *d_automata_state,
                                          unsigned short *d_chaotic_values,
                                          unsigned int *indices,
                                          size_t block_length) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= block_length)
    return;

  unsigned int val = d_automata_state[idx];
  int base_idx = idx << 1; // idx * 2

  // Split 32-bit value into two 16-bit values
  d_chaotic_values[base_idx] = static_cast<unsigned short>(val >> 16); // High 16 bits
  d_chaotic_values[base_idx + 1] = static_cast<unsigned short>(val & 0xFFFF); // Low 16 bits

  // Initialize indices for sorting
  indices[base_idx] = base_idx;
  indices[base_idx + 1] = base_idx + 1;
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
    unsigned short *cellular_automata, unsigned short *image_automata_state,
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

  unsigned short cellular_automata_value;

  size_t state_idx;
  if (tid == 0) {
    state_idx = img_dimensions.cols +
                blockIdx.x; // Unique index for extra seed to avoid race
    *c_seed = d_seeds[state_idx];
    // If d_chaotic_values_for_permutation is not null, it's the first call
    // (transition/permutation). We use image_automata_state[0] (the initialized
    // hash) as the collective seed.
    cellular_automata_value = d_chaotic_values_for_permutation != nullptr
                                  ? image_automata_state[0]
                                  : image_automata_state[blockIdx.x];
  } else {
    state_idx = x - (blockIdx.x + 1);
    *c_seed = d_seeds[state_idx];
    cellular_automata_value = cellular_automata[state_idx];
  }

  for (size_t step = 0; step < total_steps; ++step) {
    *c_seed = chaotic_function(*c_seed, r);
    __syncthreads(); // To avoid race conditions

    *c_seed = coupled_map(*c_seed, *r_seed, *l_seed, &cellular_automata_value);
    // Write chaotic values to buffer using all threads if buffer is provided
    if (d_chaotic_values_for_permutation != nullptr) {
      // We use the generated chaotic values from all threads to populate the
      // permutation buffer valid x (0 to cols-1) contributes
      if (x < img_dimensions.cols && step >= transition_length) {
        size_t valid_step = step - transition_length;
        size_t write_idx = valid_step * img_dimensions.cols + x;
        if (write_idx < permutation_block_size) {
          d_chaotic_values_for_permutation[write_idx] = *c_seed;
        }
      }
    }

    // Normal flow generation
    if (tid != 0 && step >= transition_length) {
      size_t row = step - transition_length;
      d_flow[row * img_dimensions.cols + state_idx] =
          convertToBitStream(*c_seed);
    }
    __syncthreads();
  }

  // Store final state back to global memory (Include tid=0 to persist extra
  // seeds)
  d_seeds[state_idx] = *c_seed;
  // For normal threads, persist CA. For tid=0, we use a constant so no need to
  // save back
  if (tid != 0) {
    cellular_automata[state_idx] = cellular_automata_value;
  } else {
    image_automata_state[blockIdx.x] =
        cellular_automata_value ^ convertToBitStream(*c_seed);
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
    return; // Safety check

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