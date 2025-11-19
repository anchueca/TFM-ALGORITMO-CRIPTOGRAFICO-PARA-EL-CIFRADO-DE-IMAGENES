/**
 * @file encryption_aux.cu
 * @brief Helper functions for encryption: permutation generation, permutation
 * stages and automata helpers.
 */

#include "../include/encryption_aux.cuh"

// Implementation of permutation and automata helper functions.
__host__ unsigned int *
generate_flow_permutations(const std::vector<unsigned char> block_passwords,
                           size_t block_length, size_t num_blocks,
                           const size_t transition_length) {
  if (block_passwords.size() < num_blocks) {
    throw std::runtime_error("Insufficient passwords for blocks");
  }

  size_t total_size = num_blocks * block_length;

  unsigned char *d_passwords = nullptr;
  unsigned int *d_indices = nullptr;
  double *d_chaotic_values = nullptr;

  cudaError_t err =
      cudaMalloc(&d_passwords, num_blocks * sizeof(unsigned char));
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Flow: Failed to allocate device memory for passwords");
  }

  err = cudaMalloc(&d_indices, total_size * sizeof(unsigned int));
  if (err != cudaSuccess) {
    cudaFree(d_passwords);
    throw std::runtime_error("Failed to allocate device memory for indices");
  }

  err = cudaMalloc(&d_chaotic_values, total_size * sizeof(double));
  if (err != cudaSuccess) {
    cudaFree(d_passwords);
    cudaFree(d_indices);
    throw std::runtime_error(
        "Failed to allocate device memory for chaotic values");
  }

  // Copy
  err = cudaMemcpy(d_passwords, block_passwords.data(),
                   num_blocks * sizeof(unsigned char), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_passwords);
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);
    throw std::runtime_error("Failed to copy passwords to device");
  }

  const int threadsPerBlock = 256;
  const int numBlocks = (num_blocks + threadsPerBlock - 1) / threadsPerBlock;
  double r = 0.998;

  generate_chaotic<<<numBlocks, threadsPerBlock>>>(
      d_passwords, num_blocks, d_chaotic_values, d_indices, r, block_length,
      transition_length);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_passwords);
    cudaFree(d_chaotic_values);
    throw std::runtime_error("Kernel execution failed");
  }

  err = cudaDeviceSynchronize();

  if (err != cudaSuccess) {
    cudaFree(d_passwords);
    cudaFree(d_chaotic_values);
    throw std::runtime_error("Kernel synchronization failed");
  }

  cudaFree(d_passwords);
  cudaFree(d_chaotic_values);

  return d_indices;
}

// Generate permutations from elemental automata (implementation)
__host__ unsigned int *generate_automata_permutations(
    const std::vector<ElementalCelularAutomata *> automatas, const size_t steps,
    const size_t block_length) {

  size_t num_blocks = automatas.size();
  size_t total_size = num_blocks * block_length;

  if (automatas[0]->get_size() * num_blocks != total_size * 16)
    throw std::runtime_error(
        "Incompatible automata size (" +
        std::to_string(automatas[0]->get_size() *
                       num_blocks) // Tamaño total de los autómatas
        + ") and block length (" +
        std::to_string(total_size * 16) // Tamaño del bloque en bytes
        + ")");

  // iterate

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_blocks; i++) {
    automatas[i]->iterate(steps);
  }

  unsigned int **d_automatas = nullptr; // Array of pointers to automata states
  unsigned int *d_indices = nullptr;
  unsigned short *d_chaotic_values =
      nullptr; // Here the automata states are gonna be copied

  cudaError_t err =
      cudaMalloc(&d_automatas, num_blocks * sizeof(unsigned int *));
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate device memory for " +
                             std::to_string(num_blocks));
  }

  err = cudaMalloc(&d_indices, total_size * sizeof(unsigned int));
  if (err != cudaSuccess) {
    cudaFree(d_automatas);
    throw std::runtime_error("Failed to allocate device memory for indices");
  }

  err = cudaMalloc(&d_chaotic_values, total_size * sizeof(unsigned short));
  if (err != cudaSuccess) {
    cudaFree(d_automatas);
    cudaFree(d_indices);
    throw std::runtime_error(
        "Failed to allocate device memory for chaotic values");
  }

  cudaDeviceSynchronize();

  const unsigned int **pointers_to_automata_states =
      new const unsigned int *[num_blocks];
  for (int i = 0; i < num_blocks; i++) {
    pointers_to_automata_states[i] = automatas[i]->get_cuda_state();
  }

  err = cudaMemcpy(d_automatas, pointers_to_automata_states,
                   num_blocks * sizeof(unsigned int *), cudaMemcpyHostToDevice);
  delete[] pointers_to_automata_states;
  if (err != cudaSuccess) {
    cudaFree(d_automatas);
    cudaFree(d_indices);
    throw std::runtime_error(
        "Failed to copy device memory for automatas states");
  }

  const int threadsPerBlock = 256;
  const int numKerBlocksChaotic =
      (num_blocks * block_length + threadsPerBlock - 1) / threadsPerBlock;

  generate_automata_chaotic<<<numKerBlocksChaotic, threadsPerBlock>>>(
      d_automatas, d_chaotic_values, num_blocks, d_indices, block_length);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_automatas);
    cudaFree(d_chaotic_values);
    throw std::runtime_error("generate_automata_chaotic execution failed");
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  std::cout << "\t\t\tAutomata iterations time: " << time.count() << " s"
            << std::endl;

  const int numKerBlocksSort =
      (num_blocks + threadsPerBlock - 1) / threadsPerBlock;

  start = std::chrono::high_resolution_clock::now();
  sort_indices_by_chaotic_values_global<unsigned short>
      <<<numKerBlocksSort, threadsPerBlock>>>(d_chaotic_values, num_blocks,
                                              d_indices, block_length);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  time = end - start;
  std::cout << "\t\t\tsort_indices_by_chaotic_values_global time: "
            << time.count() << " s" << std::endl;

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_automatas);
    cudaFree(d_chaotic_values);
    throw std::runtime_error(
        "sort_indices_by_chaotic_values_global synchronization failed");
  }

  cudaFree(d_automatas);
  cudaFree(d_chaotic_values);

  return d_indices;
}

__host__ void block_phase_permutation(unsigned char *d_image,
                                      unsigned char *d_image_out,
                                      unsigned int *block_permutations,
                                      Image_dimnesions img_dimensions,
                                      size_t block_size) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(
      (img_dimensions.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (img_dimensions.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  permute_blocks_kernel<<<numBlocks, threadsPerBlock>>>(
      d_image, d_image_out, block_permutations, block_size, img_dimensions);

  cudaDeviceSynchronize();
}

__host__ void rows_and_columns_permutation(unsigned char *d_image,
                                           unsigned char *d_image_out,
                                           unsigned int *d_row_permutations,
                                           unsigned int *d_col_permutations,
                                           Image_dimnesions img_dimensions,
                                           bool inverse) {
  // Launch row/column permutation kernels (implementation)
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(
      (img_dimensions.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (img_dimensions.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  if (!inverse) {
    permute_rows_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image, d_image_out, d_row_permutations, img_dimensions);

    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("Row permutation error");
    }

    cudaDeviceSynchronize();

    permute_columns_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image_out, d_image, d_col_permutations, img_dimensions);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("Col permutation error");
    }
  } else {
    permute_columns_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image, d_image_out, d_col_permutations, img_dimensions);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("Col permutation error");
    }
    cudaDeviceSynchronize();

    permute_rows_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image_out, d_image, d_row_permutations, img_dimensions);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("Row permutation error");
    }
  }

  cudaDeviceSynchronize();
}

__host__ void flow_encrypt(D_pointers &d_pointers, Image_dimnesions img_dimensions) {


  dim3 threadsPerBlock(256);
  dim3 numBlocks((img_dimensions.cols + threadsPerBlock.x - 1) /
                 threadsPerBlock.x);
  image_xor<<<numBlocks, threadsPerBlock>>>(d_pointers.d_flow,d_pointers.d_image,img_dimensions);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("Flow encryption error");
  }
  cudaDeviceSynchronize();
}

__host__ void generate_flow_stream(D_pointers &d_pointers,
                           Image_dimnesions img_dimensions, double r){

  // Launch flow stream kernel
  dim3 threadsPerBlock(256);
  dim3 numBlocks((img_dimensions.cols + threadsPerBlock.x - 1) /
                 threadsPerBlock.x);
  keystream_generation<<<numBlocks, threadsPerBlock>>>(d_pointers,
                                                     img_dimensions, r);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("generate_flow_stream: Flow generation error");
  }
  cudaDeviceSynchronize();
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

  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("Invert permutation error");
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