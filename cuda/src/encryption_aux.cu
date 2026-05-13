/**
 * @file encryption_aux.cu
 * @brief Helper functions for encryption: permutation generation, permutation
 * stages and automata helpers.
 */

#include "../include/encryption_aux.cuh"

__host__ unsigned int *
generate_automata_permutations(unsigned int *d_automata_state,
                               const size_t block_length, bool verbose) {

  auto start_chaotic = std::chrono::high_resolution_clock::now();
  size_t num_keys = block_length * 2;

  unsigned short *d_chaotic_values = nullptr;
  unsigned int *d_indices = nullptr;

  checkCudaError(
      cudaMalloc(&d_chaotic_values, num_keys * sizeof(unsigned short)),
      "cudaMalloc failed for d_chaotic_values");

  checkCudaError(cudaMalloc(&d_indices, num_keys * sizeof(unsigned int)),
                 "cudaMalloc failed for d_indices");

  int threadsPerBlock = 256;
  int numBlocks = (block_length / 2 + threadsPerBlock - 1) / threadsPerBlock;

  generate_automata_chaotic<<<numBlocks, threadsPerBlock>>>(
      d_automata_state, d_chaotic_values, d_indices, block_length);

  checkCudaError(
      cudaDeviceSynchronize(),
      "cudaDeviceSynchronize failed after generate_automata_chaotic");

  auto end_chaotic = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_chaotic = end_chaotic - start_chaotic;

  // === TIMING 3: Batched Sort ===
  auto start_sort = std::chrono::high_resolution_clock::now();
  batched_gpu_argsort(d_chaotic_values, d_indices, 1, block_length);

  checkCudaError(cudaDeviceSynchronize(),
                 "cudaDeviceSynchronize failed after batched_gpu_argsort");

  auto end_sort = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_sort = end_sort - start_sort;

  // Print detailed timing if verbose
  if (verbose) {
    std::cout << "\t\tChaotic Generation: " << time_chaotic.count() * 1000.0
              << " ms" << std::endl;
    std::cout << "\t\tBatched Sort: " << time_sort.count() * 1000.0 << " ms"
              << std::endl;
  }

  cudaFree(d_chaotic_values);

  return d_indices;
}

__host__ void generate_permutation_block(D_pointers &d_pointers,
                                         Image_dimensions img_dimensions,
                                         EncryptionParams params) {
  size_t block_size = params.block_size * params.block_size;

  if (d_pointers.d_permutation_blocks == nullptr) {
    checkCudaError(cudaMalloc(&d_pointers.d_permutation_blocks,
                              block_size * sizeof(unsigned int)),
                   "Failed to allocate device memory for indices");
  }
  // Parallel sorting (Bitonic Sort) to generate permutation from chaotic values
  compute_permutation_device(d_pointers.d_chaotic_values_for_permutation,
                             d_pointers.d_permutation_blocks, block_size);
  inverse_permutations(d_pointers.d_permutation_blocks,
                       &d_pointers.d_permutation_blocks_inverse, block_size);
}

__host__ void inverse_permutations(unsigned int *d_permutations,
                                   unsigned int **d_permutations_inverse,
                                   size_t block_length) {

  // Correctly calculate the total memory needed in bytes.
  size_t total_bytes = block_length * sizeof(unsigned int);

  // Allocate memory for the output array on the device.
  checkCudaError(cudaMalloc(d_permutations_inverse, total_bytes),
                 "Error allocating device memory for inverse permutations");

  int threadsPerBlock = 256;
  int numBlocks = (block_length + threadsPerBlock - 1) / threadsPerBlock;

  invert_permutations_kernel<<<numBlocks, threadsPerBlock>>>(
      d_permutations, *d_permutations_inverse, block_length);

  checkCudaError(cudaDeviceSynchronize(),
                 "Error during cudaDeviceSynchronize in inverse_permutations");
}

__host__ void unstack_channels_gpu(unsigned char *d_interleaved,
                                   unsigned char *d_planar, int width,
                                   int height) {
  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  deinterleave_channels_kernel<<<grid, block>>>(d_interleaved, d_planar, width,
                                                height);

  checkCudaError(
      cudaDeviceSynchronize(),
      "Error during cudaDeviceSynchronize in deinterleave_channels_kernel");
}

__host__ void stack_channels_gpu(unsigned char *d_planar,
                                 unsigned char *d_interleaved, int width,
                                 int height) {
  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  interleave_channels_kernel<<<grid, block>>>(d_planar, d_interleaved, width,
                                              height);

  checkCudaError(
      cudaDeviceSynchronize(),
      "Error during cudaDeviceSynchronize in interleave_channels_kernel");
}

__host__ void
fused_permutation_xor(unsigned char *d_image_in, unsigned char *d_image_out,
                      unsigned char *d_flow, unsigned int *d_permutation,
                      unsigned int *d_permutation_inverse,
                      unsigned int *d_blocks, unsigned int *d_blocks_inv,
                      Image_dimensions img_dimensions, size_t block_size,
                      bool use_xor, bool inverse) {

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(
      (img_dimensions.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (img_dimensions.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  fused_permutation_xor_kernel<<<numBlocks, threadsPerBlock>>>(
      d_image_in, d_image_out, d_flow, d_permutation, d_permutation_inverse,
      d_blocks, d_blocks_inv, block_size, img_dimensions.rows, use_xor,
      inverse);

  checkCudaError(cudaDeviceSynchronize(),
                 "Fused permutation XOR kernel failed");
}

__host__ void
convert_bits_to_real(const std::vector<unsigned char> &password_segment,
                     Real **d_seeds) {

  size_t total_bytes = password_segment.size();
  size_t element_size = sizeof(unsigned int);

  if (total_bytes % element_size != 0) {
    throw std::runtime_error("Invalid length.");
  }

  size_t num_elements = total_bytes / element_size;

  checkCudaError(cudaMalloc((void **)d_seeds, total_bytes),
                 "Error cudaMalloc for seeds");

  checkCudaError(cudaMemcpy(*d_seeds, password_segment.data(), total_bytes,
                            cudaMemcpyHostToDevice),
                 "Error cudaMemcpy for seeds");

  const int threadsPerBlock = 256;
  const int gridOfBlocks =
      (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  convert_bits_to_real_kernel<<<gridOfBlocks, threadsPerBlock>>>(*d_seeds,
                                                                 num_elements);

  checkCudaError(cudaDeviceSynchronize(),
                 "Error during cudaDeviceSynchronize in convert_bits_to_real");
}

__host__ void
convert_bits_to_r_params(const std::vector<unsigned char> &password_segment,
                         Real **d_r_params) {
  // Reuse the same logic as convert_bits_to_real:
  // interpret bytes as uint32_t, normalize to [0, 1].
  // The final scaling to [3, 7] is done inline in the kernel.
  convert_bits_to_real(password_segment, d_r_params);
}

__host__ void generate_flow_stream_parallel(D_pointers &d_pointers,
                                            Image_dimensions img_dimensions,
                                            EncryptionParams params) {

  // Launch flow stream kernel
  // Optimization: Reduce threads per block to increase number of blocks
  // (occupancy) Since we have limited number of columns (e.g. 512-1024), 256
  // threads only gives ~2-4 blocks. Using 64 threads gives ~8-16 blocks,
  // utilizing more SMs. Effective threads = MAX_THREADS - 1 (tid=0 is used for
  // halo/coupling)
  int effective_threads = MAX_THREADS - 1;
  dim3 threadsPerBlock(MAX_THREADS);
  dim3 numBlocks((img_dimensions.cols + effective_threads - 1) /
                 effective_threads);

  // For permutations
  size_t block_size = params.block_size * params.block_size;

  size_t transition_length;
  Real *chaotic_values = nullptr;

  // Initialization logic for first run (checked via d_image_automata_state)
  if (d_pointers.d_image_automata_state == nullptr) {
    // Allocate chaotic values if not already done (it might be passed in or
    // allocated externally)
    if (d_pointers.d_chaotic_values_for_permutation == nullptr) {
      checkCudaError(cudaMalloc(&d_pointers.d_chaotic_values_for_permutation,
                                block_size * sizeof(Real)),
                     "Failed to allocate chaotic values");
    }

    transition_length = params.transition_length;

    // Allocate and initialize automata state
    checkCudaError(cudaMalloc(&d_pointers.d_image_automata_state,
                              numBlocks.x * sizeof(unsigned short)),
                   "Failed to allocate device memory for image automata state");

    std::vector<unsigned short> init_states(numBlocks.x, params.image_hash);

    checkCudaError(cudaMemcpy(d_pointers.d_image_automata_state,
                              init_states.data(),
                              numBlocks.x * sizeof(unsigned short),
                              cudaMemcpyHostToDevice),
                   "Failed to copy hash to device memory");
                   
    chaotic_values = d_pointers.d_chaotic_values_for_permutation;
  } else {
    // Subsequent runs
    transition_length = threadsPerBlock.x / 2;
    chaotic_values = d_pointers.d_chaotic_values_for_permutation;
  }

  // Shared memory needs to hold the block's seeds plus extra seeds
  size_t shared_mem_size = threadsPerBlock.x * sizeof(Real);

  // Single kernel launch for transition + stream generation
  keystream_generation_parallel<<<numBlocks, threadsPerBlock,
                                  shared_mem_size>>>(
      d_pointers.d_flow, d_pointers.d_seeds,
      reinterpret_cast<unsigned short *>(d_pointers.d_automata_state),
      d_pointers.d_image_automata_state, img_dimensions, d_pointers.d_r_params,
      img_dimensions.rows + transition_length, chaotic_values, block_size,
      transition_length, numBlocks.x);

  // Final synchronization to ensure all stream generation is done before
  // proceeding

  checkCudaError(
      cudaDeviceSynchronize(),
      "Error during cudaDeviceSynchronize in generate_flow_stream_parallel");

  // Global Diffusion Layer: Iterative Global Mean-Field Coupling
  // This step ensures that changes in one block propagate to all blocks in the
  // next round.
  global_seed_mix_kernel<<<1, 1>>>(d_pointers.d_seeds, img_dimensions.cols,
                                   numBlocks.x);
  checkCudaError(
      cudaDeviceSynchronize(),
      "Error during cudaDeviceSynchronize in global_seed_mix_kernel");
}
