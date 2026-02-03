/**
 * @file encryption_aux.cu
 * @brief Helper functions for encryption: permutation generation, permutation
 * stages and automata helpers.
 */

#include "../include/encryption_aux.cuh"
#include <cstdio>

__host__ unsigned int *
generate_automata_permutations(ElementalCelularAutomata *automata,
                               const size_t steps, const size_t block_length,
                               bool verbose) {

  // Validate automata size
  if (automata->get_size() != block_length * 16)
    throw std::runtime_error(
        "Incompatible automata size (" + std::to_string(automata->get_size()) +
        ") and block length (" + std::to_string(block_length * 16) + ")");

  cudaError_t err;
  // === TIMING 1: Automata Iteration ===
  auto start_iterate = std::chrono::high_resolution_clock::now();
  automata->iterate_block_level(steps);
  err = cudaDeviceSynchronize();

  if (err != cudaSuccess) {
    std::cerr << " [FATAL] cudaDeviceSynchronize failed after automata "
                 "iteration. Error: "
              << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error(
        "cudaDeviceSynchronize failed after automata iteration: " +
        std::string(cudaGetErrorString(err)));
  }

  auto end_iterate = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_iterate = end_iterate - start_iterate;

  // === TIMING 2: Chaotic Generation ===
  auto start_chaotic = std::chrono::high_resolution_clock::now();

  const uint32_t *d_automata_state = automata->get_cuda_state();

  size_t num_keys = block_length * 2;

  unsigned short *d_chaotic_values = nullptr;
  unsigned int *d_indices = nullptr;

  err = cudaMalloc(&d_chaotic_values, num_keys * sizeof(unsigned short));
  if (err != cudaSuccess)
    throw std::runtime_error("cudaMalloc failed for d_chaotic_values");

  err = cudaMalloc(&d_indices, num_keys * sizeof(unsigned int));
  if (err != cudaSuccess) {
    cudaFree(d_chaotic_values);
    throw std::runtime_error("cudaMalloc failed for d_indices");
  }

  int threadsPerBlock = 256;
  int numBlocks = (block_length / 2 + threadsPerBlock - 1) / threadsPerBlock;

  generate_automata_chaotic<<<numBlocks, threadsPerBlock>>>(
      d_automata_state, d_chaotic_values, d_indices, block_length);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    if (verbose)
      std::cout << " [DEBUG] Kernel error after generate_automata_chaotic: "
                << cudaGetErrorString(err) << std::endl;
    std::cerr << " [FATAL] Kernel failed: generate_automata_chaotic. Error: "
              << cudaGetErrorString(err) << std::endl;
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);
    throw std::runtime_error("Kernel fail: generate_automata_chaotic (" +
                             std::string(cudaGetErrorString(err)) + ")");
  }
  if (err != cudaSuccess) {
    std::cerr << " [FATAL] cudaFree failed for d_automata_ptr. Error: "
              << cudaGetErrorString(err) << std::endl;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << " [FATAL] cudaDeviceSynchronize failed after "
                 "generate_automata_chaotic. Error: "
              << cudaGetErrorString(err) << std::endl;
  }
  auto end_chaotic = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_chaotic = end_chaotic - start_chaotic;

  // === TIMING 3: Batched Sort ===
  auto start_sort = std::chrono::high_resolution_clock::now();
  batched_gpu_argsort(d_chaotic_values, d_indices, 1, block_length);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);
    throw std::runtime_error(
        "cudaDeviceSynchronize failed after batched_gpu_argsort: " +
        std::string(cudaGetErrorString(err)));
  }
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
      d_pointers.d_chaotic_values_for_permutation, 1,
      d_pointers.d_permutation_blocks, block_size);
  inverse_permutations(d_pointers.d_permutation_blocks,
                       &d_pointers.d_permutation_blocks_inverse, block_size);
}

__host__ void rows_and_columns_permutation(unsigned char *d_image,
                                           unsigned char *d_image_out,
                                           unsigned int *d_permutations,
                                           unsigned int *d_permutations_inverse,
                                           Image_dimensions img_dimensions,
                                           bool inverse) {
  // Define standard block size for 2D images
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(
      (img_dimensions.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (img_dimensions.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  cudaError_t err;
  if (!inverse) {
    // --- ENCRYPTION: Rows -> Cols ---

    // Step 1: Permute Rows (Source -> Temp)
    permute_rows_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image, d_image_out, d_permutations, img_dimensions);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      throw std::runtime_error("Row permutation kernel failed: " +
                               std::string(cudaGetErrorString(err)));
    }

    // Step 2: Permute Columns (Temp -> Source)
    permute_columns_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image_out, d_image, d_permutations_inverse, img_dimensions);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      throw std::runtime_error("Col permutation kernel failed: " +
                               std::string(cudaGetErrorString(err)));
    }

  } else {
    // --- DECRYPTION: Inverse Cols -> Inverse Rows ---
    // Order must be strictly reversed relative to encryption

    // Step 1: Inverse Permute Columns (Source -> Temp)
    permute_columns_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image, d_image_out, d_permutations_inverse, img_dimensions);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      throw std::runtime_error("Col permutation (inverse) kernel failed: " +
                               std::string(cudaGetErrorString(err)));
    }

    // Step 2: Inverse Permute Rows (Temp -> Source)
    permute_rows_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image_out, d_image, d_permutations, img_dimensions);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      throw std::runtime_error("Row permutation (inverse) kernel failed: " +
                               std::string(cudaGetErrorString(err)));
    }
  }
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
                                   size_t block_length) {

  // Correctly calculate the total memory needed in bytes.
  size_t total_bytes = block_length * sizeof(unsigned int);
  cudaError_t err;

  // Allocate memory for the output array on the device.
  err = cudaMalloc(d_permutations_inverse, total_bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Error allocating device memory for inverse permutations");
  }

  int threadsPerBlock = 256;
  int numBlocks = (block_length + threadsPerBlock - 1) / threadsPerBlock;

  invert_permutations_kernel<<<numBlocks, threadsPerBlock>>>(
      d_permutations, *d_permutations_inverse, block_length);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(*d_permutations_inverse);
    throw std::runtime_error(
        std::string("Kernel launch error in inverse_permutations: ") +
        cudaGetErrorString(err));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(*d_permutations_inverse);
    throw std::runtime_error(
        std::string(
            "Error during cudaDeviceSynchronize in inverse_permutations: ") +
        cudaGetErrorString(err));
  }
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
    throw std::runtime_error("Error cudaMalloc for seeds," +
                             std::string(cudaGetErrorString(err)));

  err = cudaMemcpy(*d_seeds, password_segment.data(), total_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(*d_seeds);
    throw std::runtime_error("Error cudaMemcpy for seeds," +
                             std::string(cudaGetErrorString(err)));
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

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(*d_seeds);
    throw std::runtime_error(
        "Error during cudaDeviceSynchronize in convert_bits_to_real");
  }
}

__host__ void generate_flow_stream_parallel(D_pointers &d_pointers,
                                            Image_dimensions img_dimensions,
                                            EncryptionParams params) {

  // Launch flow stream kernel
  int max_threads = 256;
  // Effective threads = max_threads - 1 (tid=0 is used for halo/coupling)
  int effective_threads = max_threads - 1;
  dim3 threadsPerBlock(max_threads);
  dim3 numBlocks((img_dimensions.cols + effective_threads - 1) /
                 effective_threads);

  // For permutations
  size_t block_size = params.block_size * params.block_size;

  size_t transition_length;
  Real *chaotic_values;

  // Initialization logic for first run (checked via d_image_automata_state)
  cudaError_t err;
  if (d_pointers.d_image_automata_state == nullptr) {
    // Allocate chaotic values if not already done (it might be passed in or
    // allocated externally)
    if (d_pointers.d_chaotic_values_for_permutation == nullptr) {
      err = cudaMalloc(&d_pointers.d_chaotic_values_for_permutation,
                       block_size * sizeof(Real));
      if (err != cudaSuccess)
        throw std::runtime_error("Failed to alloc chaotic values");
    }

    transition_length = params.transition_length;
    chaotic_values = d_pointers.d_chaotic_values_for_permutation;

    // Allocate and initialize automata state
    err = cudaMalloc(&d_pointers.d_image_automata_state,
                     numBlocks.x * sizeof(unsigned short));
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate device memory for image automata state");
    }
    std::vector<unsigned short> init_states(numBlocks.x, params.image_hash);
    err = cudaMemcpy(d_pointers.d_image_automata_state, init_states.data(),
                     numBlocks.x * sizeof(unsigned short),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to copy hash to device memory");
    }
  } else {
    // Subsequent runs
    transition_length = threadsPerBlock.x / 2;
    chaotic_values = nullptr;
  }

  // Shared memory needs to hold the block's seeds plus extra seeds
  size_t shared_mem_size = threadsPerBlock.x * sizeof(Real);

  // Single kernel launch for transition + stream generation
  keystream_generation_parallel<<<numBlocks, threadsPerBlock,
                                  shared_mem_size>>>(
      d_pointers.d_flow, d_pointers.d_seeds,
      reinterpret_cast<unsigned short *>(d_pointers.d_automata_state),
      d_pointers.d_image_automata_state, img_dimensions, params.chaos_parameter,
      img_dimensions.rows + transition_length, chaotic_values, block_size,
      transition_length, numBlocks.x);

  // Final synchronization to ensure all stream generation is done before
  // proceeding
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "generate_flow_stream_parallel: Kernel launch error");
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(d_pointers.d_image_automata_state);
    throw std::runtime_error(
        "Error during cudaDeviceSynchronize in generate_flow_stream_parallel");
  }

  // Global Diffusion Layer: Iterative Global Mean-Field Coupling
  // This step ensures that changes in one block propagate to all blocks in the
  // next round.
  global_seed_mix_kernel<<<1, 1>>>(d_pointers.d_seeds, img_dimensions.cols,
                                   numBlocks.x);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(d_pointers.d_image_automata_state);
    throw std::runtime_error(
        "Error during cudaDeviceSynchronize in global_seed_mix_kernel");
  }
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