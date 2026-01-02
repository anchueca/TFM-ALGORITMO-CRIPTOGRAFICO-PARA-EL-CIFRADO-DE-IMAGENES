/**
 * @file encryption_aux.cu
 * @brief Helper functions for encryption: permutation generation, permutation
 * stages and automata helpers.
 */

#include "../include/encryption_aux.cuh"

__host__ unsigned int *
generate_automata_permutations(ElementalCelularAutomata *automata,
                               const size_t steps, const size_t block_length,
                               bool verbose) {

  // Validate automata size
  if (automata->get_size() != block_length * 16)
    throw std::runtime_error(
        "Incompatible automata size (" + std::to_string(automata->get_size()) +
        ") and block length (" + std::to_string(block_length * 16) + ")");

  // === TIMING 1: Automata Iteration ===
  auto start_iterate = std::chrono::high_resolution_clock::now();
  automata->iterate_block_level(steps);
  cudaDeviceSynchronize();
  auto end_iterate = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_iterate = end_iterate - start_iterate;

  // Allocate device memory
  unsigned int *d_indices = nullptr;
  unsigned short *d_chaotic_values = nullptr;

  cudaError_t err = cudaMalloc(&d_indices, block_length * sizeof(unsigned int));
  if (err != cudaSuccess)
    throw std::runtime_error("Alloc failed: d_indices");

  err = cudaMalloc(&d_chaotic_values, block_length * sizeof(unsigned short));
  if (err != cudaSuccess) {
    cudaFree(d_indices);
    throw std::runtime_error("Alloc failed: d_chaotic_values");
  }

  // === TIMING 2: Chaotic Generation ===
  auto start_chaotic = std::chrono::high_resolution_clock::now();

  // Optimize thread/block configuration based on block_length
  int threadsPerBlock;
  if (block_length <= 256) {
    // For small sizes, use nearest power of 2
    threadsPerBlock = 1 << (32 - __builtin_clz(block_length - 1));
    threadsPerBlock = std::min(threadsPerBlock, 256);
  } else if (block_length <= 1024) {
    threadsPerBlock = 512; // Better occupancy for medium sizes
  } else {
    threadsPerBlock = 1024; // Maximum for large sizes
  }
  const int numBlocks = (block_length + threadsPerBlock - 1) / threadsPerBlock;

  // Allocate single-element array for pointer to automata state
  unsigned int **d_automata_ptr = nullptr;
  unsigned int *h_automata_state =
      const_cast<unsigned int *>(automata->get_cuda_state());

  err = cudaMalloc(&d_automata_ptr, sizeof(unsigned int *));
  if (err != cudaSuccess) {
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);
    throw std::runtime_error("Alloc failed: d_automata_ptr");
  }

  err = cudaMemcpy(d_automata_ptr, &h_automata_state, sizeof(unsigned int *),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);
    cudaFree(d_automata_ptr);
    throw std::runtime_error("Memcpy failed: automata pointer");
  }

  generate_automata_chaotic<<<numBlocks, threadsPerBlock>>>(
      d_automata_ptr, d_chaotic_values, 1, d_indices, block_length);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);
    cudaFree(d_automata_ptr);
    throw std::runtime_error("Kernel fail: generate_automata_chaotic");
  }

  cudaFree(d_automata_ptr); // No longer needed after kernel execution

  cudaDeviceSynchronize();
  auto end_chaotic = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_chaotic = end_chaotic - start_chaotic;

  // === TIMING 3: Batched Sort ===
  auto start_sort = std::chrono::high_resolution_clock::now();
  batched_gpu_argsort(d_chaotic_values, d_indices, 1, block_length);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);
    throw std::runtime_error("Kernel fail: batched_gpu_argsort");
  }

  cudaDeviceSynchronize();
  auto end_sort = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_sort = end_sort - start_sort;

  // Print detailed timing if verbose
  if (verbose) {
    std::cout << "\t\tAutomata Iteration: " << time_iterate.count() * 1000.0f
              << " ms" << std::endl;
    std::cout << "\t\tChaotic Generation: " << time_chaotic.count() * 1000.0f
              << " ms (blocks=" << numBlocks << ", threads=" << threadsPerBlock
              << ")" << std::endl;
    std::cout << "\t\tBatched Sort: " << time_sort.count() * 1000.0f << " ms"
              << std::endl;
  }

  cudaFree(d_chaotic_values);

  return d_indices;
}

__host__ void block_phase_permutation(unsigned char *d_image,
                                      unsigned char *d_image_out,
                                      unsigned int *permutation,
                                      unsigned int *permutation_inverse,
                                      Image_dimensions img_dimensions,
                                      size_t block_size) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(
      (img_dimensions.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (img_dimensions.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  permute_blocks_kernel_simple<<<numBlocks, threadsPerBlock>>>(
      d_image, d_image_out, permutation, permutation_inverse, block_size,
      img_dimensions);
  cudaDeviceSynchronize();
}

__host__ void generate_permutation_block(D_pointers &d_pointers,
                                         Image_dimensions img_dimensions,
                                         EncryptionParams params) {
  size_t block_size = params.block_size * params.block_size;

  if (d_pointers.d_permutation_blocks == nullptr) {
    cudaError_t err = cudaMalloc(&d_pointers.d_permutation_blocks,
                                 block_size * sizeof(unsigned int));
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate device memory for indices");
    }
  }
  dim3 threadsPerBlock(256);
  dim3 numBlocks((img_dimensions.cols + threadsPerBlock.x) / threadsPerBlock.x);
  sort_indices_by_chaotic_values_global<<<1, 1>>>(
      d_pointers.d_chaotic_values, 1, d_pointers.d_permutation_blocks,
      block_size);
  inverse_permutations(d_pointers.d_permutation_blocks,
                       &d_pointers.d_permutation_blocks_inverse, block_size, 1);
}

__host__ void rows_and_columns_permutation(unsigned char *d_image,
                                           unsigned char *d_image_out,
                                           unsigned int *d_row_permutations,
                                           unsigned int *d_col_permutations,
                                           Image_dimensions img_dimensions,
                                           bool inverse) {
  // Define standard block size for 2D images
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(
      (img_dimensions.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (img_dimensions.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  if (!inverse) {
    // --- ENCRYPTION: Rows -> Cols ---

    // Step 1: Permute Rows (Source -> Temp)
    permute_rows_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image, d_image_out, d_row_permutations, img_dimensions);

    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("Row permutation kernel failed");
    }

    // Step 2: Permute Columns (Temp -> Source)
    permute_columns_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image_out, d_image, d_col_permutations, img_dimensions);

    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("Col permutation kernel failed");
    }

  } else {
    // --- DECRYPTION: Inverse Cols -> Inverse Rows ---
    // Order must be strictly reversed relative to encryption

    // Step 1: Inverse Permute Columns (Source -> Temp)
    permute_columns_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image, d_image_out, d_col_permutations, img_dimensions);

    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("Col permutation (inverse) kernel failed");
    }

    // Step 2: Inverse Permute Rows (Temp -> Source)
    permute_rows_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image_out, d_image, d_row_permutations, img_dimensions);

    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("Row permutation (inverse) kernel failed");
    }
  }

  // Ensure all GPU tasks are finished before returning to host code
  cudaDeviceSynchronize();
}

__host__ void flow_encrypt(D_pointers &d_pointers,
                           Image_dimensions img_dimensions) {

  dim3 threadsPerBlock(16, 16);

  dim3 numBlocks(
      (img_dimensions.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (img_dimensions.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  image_xor<<<numBlocks, threadsPerBlock>>>(d_pointers.d_flow,
                                            d_pointers.d_image, img_dimensions);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::string errorMsg = "Flow encryption kernel launch error: ";
    errorMsg += cudaGetErrorString(err);
    throw std::runtime_error(errorMsg);
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::string errorMsg = "Flow encryption execution error: ";
    errorMsg += cudaGetErrorString(err);
    throw std::runtime_error(errorMsg);
  }
}

__host__ void inverse_permutations(unsigned int *d_permutations,
                                   unsigned int **d_permutations_inverse,
                                   size_t block_length,
                                   size_t num_permutations) {

  // Correctly calculate the total memory needed in bytes.
  size_t total_elements = block_length * num_permutations;
  size_t total_bytes = total_elements * sizeof(unsigned int);

  // Allocate memory for the output array on the device.
  cudaMalloc(d_permutations_inverse, total_bytes);

  // Configure the kernel launch: one block per permutation.
  dim3 threadsPerBlock(std::min(static_cast<size_t>(512), block_length));
  dim3 gridOfBlocks(num_permutations);

  invert_permutations_kernel<<<gridOfBlocks, threadsPerBlock>>>(
      d_permutations, *d_permutations_inverse, block_length, num_permutations);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(*d_permutations_inverse);
    throw std::runtime_error(
        std::string("Kernel launch error in inverse_permutations: ") +
        cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}

__host__ void unstack_channels_gpu(unsigned char *d_interleaved,
                                   unsigned char *d_planar, int width,
                                   int height) {
  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  deinterleave_channels_kernel<<<grid, block>>>(d_interleaved, d_planar, width,
                                                height);

  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error("Launch error: deinterleave_channels_kernel");
  }
  cudaDeviceSynchronize();
}

__host__ void stack_channels_gpu(unsigned char *d_planar,
                                 unsigned char *d_interleaved, int width,
                                 int height) {
  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  interleave_channels_kernel<<<grid, block>>>(d_planar, d_interleaved, width,
                                              height);

  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error("Launch error: interleave_channels_kernel");
  }
  cudaDeviceSynchronize();
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

  cudaError_t err = cudaMalloc((void **)d_seeds, total_bytes);
  if (err != cudaSuccess)
    throw std::runtime_error("Error cudaMalloc");

  err = cudaMemcpy(*d_seeds, password_segment.data(), total_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(*d_seeds);
    throw std::runtime_error("Error cudaMemcpy");
  }

  const int threadsPerBlock = 256;
  const int gridOfBlocks =
      (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  convert_bits_to_real_kernel<<<gridOfBlocks, threadsPerBlock>>>(*d_seeds,
                                                                 num_elements);

  if (cudaGetLastError() != cudaSuccess) {
    cudaFree(*d_seeds);
    throw std::runtime_error("Error en kernel convert_bits_to_real");
  }

  cudaDeviceSynchronize();
}

__host__ void generate_flow_stream_parallel(D_pointers &d_pointers,
                                            Image_dimensions img_dimensions,
                                            EncryptionParams params) {

  // Launch flow stream kernel
  dim3 threadsPerBlock(256);
  dim3 numBlocks((img_dimensions.cols + threadsPerBlock.x) / threadsPerBlock.x);

  // For permutations
  size_t block_size = params.block_size * params.block_size;

  size_t transition_length;
  Real *chaotic_values;

  cudaError_t err;
  if (d_pointers.d_chaotic_values == nullptr) {
    err = cudaMalloc(&d_pointers.d_chaotic_values, block_size * sizeof(Real));
    if (err != cudaSuccess) {
      cudaFree(d_pointers.d_permutation_blocks);
      throw std::runtime_error(
          "Failed to allocate device memory for chaotic values");
    }
    // First time only transition is computed
    transition_length = params.transition_length;
    chaotic_values = d_pointers.d_chaotic_values;
  } else {
    // First transition is already computed
    transition_length = 0;
    chaotic_values = nullptr;
  }

  // Shared memory needs to hold the block's seeds plus extra seeds
  size_t shared_mem_size = threadsPerBlock.x * sizeof(Real);

  // Single kernel launch for transition + stream generation
  keystream_generation_parallel<<<numBlocks, threadsPerBlock,
                                  shared_mem_size>>>(
      d_pointers.d_flow, d_pointers.d_seeds,
      reinterpret_cast<unsigned short *>(d_pointers.d_automata_state),
      img_dimensions, params.chaos_parameter,
      img_dimensions.rows + transition_length, chaotic_values, block_size,
      transition_length, numBlocks.x);

  // Final synchronization to ensure all stream generation is done before
  // proceeding
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "generate_flow_stream_parallel: Kernel launch error");
  }

  cudaDeviceSynchronize();
}

__global__ void sort_indices_by_chaotic_values_global(Real *d_chaotic_values,
                                                      size_t num_blocks,
                                                      unsigned int *indices,
                                                      size_t block_length) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= (int)num_blocks)
    return;
  int base_idx = idx * (int)block_length;

  for (int i = 0; i < block_length; i++) { // Create indices
    indices[base_idx + i] = i;
  }

  sort_indices_by_chaotic_values(base_idx, d_chaotic_values, indices,
                                 block_length);
}