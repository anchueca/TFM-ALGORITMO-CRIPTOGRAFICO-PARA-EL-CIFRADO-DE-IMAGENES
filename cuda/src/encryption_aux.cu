/**
 * @file encryption_aux.cu
 * @brief Helper functions for encryption: permutation generation, permutation
 * stages and automata helpers.
 */

#include "../include/encryption_aux.cuh"
#include <cstddef>

__host__ unsigned int *generate_automata_permutations(
    const std::vector<ElementalCelularAutomata *> automatas, const size_t steps,
    const size_t block_length, bool verbose) {

  size_t num_blocks = automatas.size();
  size_t total_size = num_blocks * block_length;

  if (automatas[0]->get_size() * num_blocks != total_size * 16)
    throw std::runtime_error(
        "Incompatible automata size (" +
        std::to_string(automatas[0]->get_size() * num_blocks) +
        ") and block length (" + std::to_string(total_size * 16) + ")");

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_blocks; i++) {
    automatas[i]->iterate(steps);
  }

  // Allocations
  unsigned int **d_automatas = nullptr;
  unsigned int *d_indices = nullptr;
  unsigned short *d_chaotic_values = nullptr;

  cudaError_t err =
      cudaMalloc(&d_automatas, num_blocks * sizeof(unsigned int *));
  if (err != cudaSuccess)
    throw std::runtime_error("Alloc failed: d_automatas");

  err = cudaMalloc(&d_indices, total_size * sizeof(unsigned int));
  if (err != cudaSuccess) {
    cudaFree(d_automatas);
    throw std::runtime_error("Alloc failed: d_indices");
  }

  err = cudaMalloc(&d_chaotic_values, total_size * sizeof(unsigned short));
  if (err != cudaSuccess) {
    cudaFree(d_automatas);
    cudaFree(d_indices);
    throw std::runtime_error("Alloc failed: d_chaotic_values");
  }

  const unsigned int **pointers_to_automata_states =
      new const unsigned int *[num_blocks];
  for (int i = 0; i < num_blocks; i++) {
    pointers_to_automata_states[i] = automatas[i]->get_cuda_state();
  }

  err = cudaMemcpy(d_automatas, pointers_to_automata_states,
                   num_blocks * sizeof(unsigned int *), cudaMemcpyHostToDevice);
  delete[] pointers_to_automata_states;
  if (err != cudaSuccess)
    throw std::runtime_error("Memcpy failed: automata pointers");

  // Generate chaotic values
  const int threadsPerBlock = 256;
  const int numKerBlocksChaotic =
      (total_size + threadsPerBlock - 1) / threadsPerBlock;

  generate_automata_chaotic<<<numKerBlocksChaotic, threadsPerBlock>>>(
      d_automatas, d_chaotic_values, num_blocks, d_indices, block_length);

  err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error("Kernel fail: generate_automata_chaotic");

  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  if (verbose)
    std::cout << "\t\tAutomata & Gen time: " << time.count() * 1000.0f << " ms"
              << std::endl;

  // Short
  start = std::chrono::high_resolution_clock::now();

  batched_gpu_argsort(d_chaotic_values, d_indices, num_blocks, block_length);

  err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error("Kernel fail: batched_gpu_argsort");

  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  time = end - start;
  if (verbose)
    std::cout << "\t\tBatched Sort time: " << time.count() * 1000.0f << " ms"
              << std::endl;

  cudaFree(d_automatas);
  cudaFree(d_chaotic_values);

  return d_indices;
}

__host__ void block_phase_permutation_simple(unsigned char *d_image,
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
                                 block_size * params.num_blocks_permutations *
                                     sizeof(unsigned int));
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate device memory for indices");
    }
  }
  dim3 threadsPerBlock(256);
  dim3 numBlocks((img_dimensions.cols + params.num_blocks_permutations +
                  threadsPerBlock.x - 1) /
                 threadsPerBlock.x);
  sort_indices_by_chaotic_values_global<<<params.num_blocks_permutations, 1>>>(
      d_pointers.d_chaotic_values, params.num_blocks_permutations,
      d_pointers.d_permutation_blocks, block_size);
  inverse_permutations(d_pointers.d_permutation_blocks,
                       &d_pointers.d_permutation_blocks_inverse, block_size,
                       params.num_blocks_permutations);
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

__host__ const std::vector<ElementalCelularAutomata *> createElementalAutomata(
    const std::vector<std::vector<unsigned char>> &password_segments,
    size_t num_blocks, size_t block_size, size_t precision_level) {

  // Create automata instances from password segments (implementation)
  std::vector<ElementalCelularAutomata *> container(num_blocks);

  const size_t byte_size = block_size * precision_level;

  for (size_t i = 0; i < num_blocks; ++i) {
    unsigned int *cuda_pointer = nullptr;

    cudaError_t err = cudaMalloc(&cuda_pointer, byte_size);
    if (err != cudaSuccess) {
      std::cerr << "CUDA memory allocation error: " << cudaGetErrorString(err)
                << std::endl;
      return {};
    }

    const unsigned char *src_ptr = password_segments[2].data() + i * byte_size;
    err = cudaMemcpy(cuda_pointer, src_ptr, byte_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cerr << "CUDA memcpy error when copying initial automata state: "
                << cudaGetErrorString(err) << std::endl;
      return {};
    }

    container[i] =
        new ElementalCelularAutomata(cuda_pointer, byte_size * 8, 30);
  }
  return container;
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