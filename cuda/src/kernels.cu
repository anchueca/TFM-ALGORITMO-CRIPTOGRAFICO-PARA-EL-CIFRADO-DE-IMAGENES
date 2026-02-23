/**
 * @file kernels.cu
 * @brief CUDA kernels used by the encryption pipeline: flow generator and
 * permutation kernels.
 */

#include "../include/kernels.cuh"

__device__ __forceinline__ CoupledResult coupled_map(Real c_next, Real *r_next,
                                                    Real *l_next,
                                                    unsigned short ca_state) {

  // Evolve CA by-value to avoid requiring callers to pass the address of
  // a local variable (which would force a spill to local memory).
  unsigned short evolved = evolve_16bit_isolated(ca_state, 30, 1);

  // Extract weights from the evolved CA state
  Real v1 = static_cast<Real>((evolved >> 8) & 0xFF) / 255.0;
  Real v2 = static_cast<Real>(evolved & 0xFF) / 255.0;

  Real c_influence = v1;
  Real rest = (Real)1.0 - v1;
  Real r_influence = rest * v2;
  Real l_influence = rest * ((Real)1.0 - v2);

  CoupledResult res;
  res.mixed = (c_next * c_influence) + (*r_next * r_influence) +
              (*l_next * l_influence);
  res.new_ca = evolved;
  return res;
}

/**
 * @brief Unified kernel that performs row, column, and block permutations
 * along with an optional XOR diffusion step.
 *
 * This kernel uses a "Gather" approach to ensure coalesced writes to global
 * memory. The sequence of operations (Rows -> Cols -> Blocks) is composed
 * such that we calculate the final source pixel for each destination pixel.
 *
 * Gather Logic (Encryption Order):
 *  1. Apply Block Permutation to get intermediate source (src_x_b, src_y_b).
 *  2. Apply Row and Column Permutations to (src_x_b, src_y_b) to get
 *     final source indices.
 *  3. (Optional) XOR the value with keystream.
 */
__global__ void fused_permutation_xor_kernel(
    const unsigned char *__restrict__ image_in,
    unsigned char *__restrict__ image_out,
    const unsigned char *__restrict__ flow,
    const unsigned int *__restrict__ permutation,
    const unsigned int *__restrict__ permutation_inverse,
    const unsigned int *__restrict__ perm_blocks,
    const unsigned int *__restrict__ perm_blocks_inv, size_t block_size,
    const size_t img_dim, bool use_xor, bool inverse_order) {

  // Global thread coordinates (Destination pixel mapping)
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Boundary check to prevent out-of-bounds memory access
  if (x >= img_dim || y >= img_dim)
    return;

  // Unified linear index for fully coalesced memory writes/reads
  const int idx = y * img_dim + x;

  // CASE 1: Flow Permutation (XOR) OR Forward Image Permutation
  if (use_xor || !inverse_order) {
    for(int i=0;i<2;i++){
      // We need to find the source pixel that maps to (x, y).
      // Since the forward order is Rows -> Cols -> Blocks,
      // we must undo them in reverse order: Blocks^-1 -> Cols^-1 -> Rows^-1.

      // 1. Undo Blocks
      int bx = x / block_size;
      int by = y / block_size;
      int lx = x - bx * block_size;
      int ly = y - by * block_size;

      // Checkerboard pattern: Even sum uses perm_blocks in forward mode.
      // Therefore, to UNDO an even block, we use perm_blocks_inv.
      bool is_odd_parity = ((bx + by) & 1);
      const unsigned int *gather_block_undo =
          is_odd_parity ? perm_blocks : perm_blocks_inv;
      unsigned int pi = gather_block_undo[ly * block_size + lx];

      int temp_x = bx * block_size + (pi % block_size);
      int temp_y = by * block_size + (pi / block_size);

      // 2 & 3. Undo Columns and Rows
      // To undo the mappings, we use the opposite permutation array.
      x = permutation[temp_x];         // Undo Column permutation
      y = permutation_inverse[temp_y]; // Undo Row permutation
    }
    if (use_xor) {
      // Image remains static; apply XOR with the dynamically permuted flow key
      image_out[idx] = image_in[idx] ^ flow[y * img_dim + x];
    } else {
      // Permute the image by copying the source pixel to the target destination
      image_out[idx] = image_in[y * img_dim + x];
    }
    
  }
  // CASE 2: Image Permutation Decryption (No XOR, inverse mode)
  else {
    for(int i=0;i<2;i++){
      // We want to find where the original pixel at (x,y) ended up in the
      // ciphered image. We apply the forward transformation route: Rows -> Cols
      // -> Blocks.

      // 1 & 2. Apply Rows and Columns
      int temp_y = permutation[y];         // Apply Row permutation
      int temp_x = permutation_inverse[x]; // Apply Column permutation

      // 3. Apply Blocks (use subtraction to compute modulo)
      int bx = temp_x / block_size;
      int by = temp_y / block_size;
      int lx = temp_x - bx * block_size;
      int ly = temp_y - by * block_size;

      // Apply forward block permutation using the checkerboard logic
      bool is_odd_parity = ((bx + by) & 1);
      const unsigned int *gather_block_fwd =
          is_odd_parity ? perm_blocks_inv : perm_blocks;
      unsigned int pi = gather_block_fwd[ly * block_size + lx];

      x = bx * block_size + (pi % block_size);
      y = by * block_size + (pi / block_size);
    }
    // Retrieve the ciphered pixel and restore it to its original unencrypted
    // position
    image_out[idx] = image_in[y * img_dim + x];
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
  d_chaotic_values[base_idx] =
      static_cast<unsigned short>(val >> 16); // High 16 bits
  d_chaotic_values[base_idx + 1] =
      static_cast<unsigned short>(val & 0xFFFF); // Low 16 bits

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
    unsigned short *__restrict__ cellular_automata,
    unsigned short *__restrict__ image_automata_state,
    Image_dimensions img_dimensions, Real r, const size_t total_steps,
    Real *__restrict__ d_chaotic_values_for_permutation,
    size_t permutation_block_size, size_t transition_length, size_t numBlocks) {

  // Shared size is determined at kernel launch
  extern __shared__ char shared_mem[];
  Real *s_seeds = reinterpret_cast<Real *>(shared_mem);

  // Thread ID
  const int tid = threadIdx.x;
  const int x = blockIdx.x * blockDim.x + tid;

  const size_t cols_and_blocks = img_dimensions.cols + numBlocks;

  // Correct block_length for the last block
  const size_t current_block_length =
      (blockIdx.x < numBlocks - 1)
          ? blockDim.x
          : (cols_and_blocks - (numBlocks - 1) * blockDim.x);

  const bool is_active = (tid < current_block_length);

  Real *c_seed = nullptr;
  Real *r_seed = nullptr;
  Real *l_seed = nullptr;

  unsigned short cellular_automata_value;
  size_t state_idx;
  Real next_val;

  if (is_active) {
    c_seed = &s_seeds[tid];
    r_seed = &s_seeds[tid == 0 ? current_block_length - 1 : tid - 1];
    l_seed = &s_seeds[tid == current_block_length - 1 ? 0 : tid + 1];

    if (tid == 0) { // Special seed for perturbation
      state_idx = img_dimensions.cols + blockIdx.x;
      *c_seed = d_seeds[state_idx];
      cellular_automata_value = d_chaotic_values_for_permutation != nullptr
                                    ? image_automata_state[0]
                                    : image_automata_state[blockIdx.x];
    } else {
      state_idx = x - (blockIdx.x + 1);
      *c_seed = d_seeds[state_idx];
      cellular_automata_value = cellular_automata[state_idx];
    }
    next_val = *c_seed;
  }

  for (size_t step = 0; step < total_steps; ++step) {
    if (is_active)
      next_val = chaotic_function(next_val, r);

    // Initial synchronization to ensure shared memory is populated (at first
    // iteration) or updated (in subsequent iterations) before any thread reads
    // from it
    __syncthreads();

    if (is_active)
      *c_seed = next_val;

    // Ensure all threads have updated their s_seeds
    __syncthreads();

    if (is_active) {
      {
        CoupledResult _cr = coupled_map(next_val, r_seed, l_seed,
                                         cellular_automata_value);
        next_val = _cr.mixed;
        cellular_automata_value = _cr.new_ca;
      }

      // Write chaotic values to buffer using all threads if buffer is provided
      if (d_chaotic_values_for_permutation != nullptr) {
        if (tid != 0 && state_idx < img_dimensions.cols &&
            step >= transition_length) {
          size_t valid_step = step - transition_length;
          size_t write_idx = valid_step * img_dimensions.cols + state_idx;
          if (write_idx < permutation_block_size) {
            d_chaotic_values_for_permutation[write_idx] = next_val;
          }
        }
      }

      // Normal flow generation
      if (tid != 0 && step >= transition_length) {
        size_t row = step - transition_length;
        d_flow[row * img_dimensions.cols + state_idx] =
            convertToBitStream(next_val);
      }
    }
  }

  if (is_active) {
    // Store final state back to global memory
    d_seeds[state_idx] = *c_seed;
    if (tid != 0) {
      cellular_automata[state_idx] = cellular_automata_value;
    } else {
      image_automata_state[blockIdx.x] =
          cellular_automata_value ^ convertToBitStream(*c_seed);
    }
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