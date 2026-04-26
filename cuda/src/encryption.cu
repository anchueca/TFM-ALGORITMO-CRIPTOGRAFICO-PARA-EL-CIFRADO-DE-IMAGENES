#include "../include/encryption.cuh"

// =================================================================================
//                                  HELPER FUNCTIONS
// =================================================================================

void print_encryption_report(const cv::Mat &image,
                             const Image_dimensions &img_dimensions,
                             const EncryptionParams &params, bool encrypt) {
  const size_t num_blocks_per_row =
      img_dimensions.rows / params.block_size +
      (img_dimensions.rows % params.block_size != 0);
  const size_t num_blocks_per_col =
      img_dimensions.cols / params.block_size +
      (img_dimensions.cols % params.block_size != 0);
  const size_t num_blocks = num_blocks_per_row * num_blocks_per_col;

  std::cout << "\n============================================================"
            << std::endl;
  std::cout << "               ENCRYPTION CONFIGURATION REPORT              "
            << std::endl;
  std::cout << "============================================================"
            << std::endl;
  std::cout << " [IMAGE PROPERTIES]" << std::endl;
  std::cout << "  " << std::left << std::setw(25)
            << "Operation Mode:" << (encrypt ? "ENCRYPTION" : "DECRYPTION")
            << std::endl;
  std::cout << "  " << std::left << std::setw(25) << "Original Format:"
            << (image.channels() == 3 ? "Color (RGB)" : "Grayscale (1-CH)")
            << std::endl;
  std::cout << "  " << std::left << std::setw(25)
            << "Processing Dims:" << img_dimensions.cols << " x "
            << img_dimensions.rows << " px" << std::endl;
  std::cout << "\n [ALGORITHM SETTINGS]" << std::endl;
  std::cout << "  " << std::left << std::setw(25) << "Rounds:" << params.rounds
            << std::endl;
  std::cout << "  " << std::left << std::setw(25)
            << "Block Size:" << params.block_size << " px" << std::endl;
  std::cout << "  " << std::left << std::setw(25)
            << "CA Evolution Steps:" << params.automata_steps << std::endl;
  std::cout << "  " << std::left << std::setw(25)
            << "CML Transition Period:" << params.transition_length
            << std::endl;
  std::cout << "\n [GRID ARCHITECTURE]" << std::endl;
  std::cout << "  " << std::left << std::setw(25)
            << "Grid Layout:" << num_blocks_per_col << " (cols) x "
            << num_blocks_per_row << " (rows)" << std::endl;
  std::cout << "  " << std::left << std::setw(25)
            << "Total Blocks:" << num_blocks << std::endl;
  std::cout << "============================================================\n"
            << std::endl;
}

void setup_permutations(D_pointers &d_pointers,
                        std::vector<std::vector<unsigned char>> &password,
                        const Image_dimensions &img_dimensions,
                        const EncryptionParams &params, bool verbose) {
  if (verbose)
    std::cout << " > Generating Permutations..." << std::endl;

  if (verbose)
    std::cout << "\t(Processing Cols Automata...)" << std::endl;
  ElementalCelularAutomata automata(password[0], img_dimensions.cols * 2 * 8,
                                    30);
  d_pointers.d_permutation_vector = generate_automata_permutations(
      &automata, params.automata_steps, img_dimensions.cols, verbose);

  // Copy CA state to persistent buffer
  size_t state_size_in_bytes = automata.get_size_in_bytes();
  cudaError_t err;

  err = cudaMalloc(&d_pointers.d_automata_state, state_size_in_bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Error allocating device memory for automata state");
  }
  err = cudaMemcpy(d_pointers.d_automata_state, automata.get_cuda_state(),
                   state_size_in_bytes, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_pointers.d_automata_state);
    throw std::runtime_error("Error copying automata state to device memory");
  }

  // 2. Compute P_inv (stored in d_permutation_vector_inverse)
  if (verbose)
    std::cout << " > Calculating Inverse Permutation P_inv..." << std::endl;

  inverse_permutations(d_pointers.d_permutation_vector,
                       &d_pointers.d_permutation_vector_inverse,
                       img_dimensions.cols);
}

void allocate_and_transfer_image(D_pointers &d_pointers, cv::Mat &image,
                                 const EncryptionParams &params, bool verbose) {

  if (verbose)
    std::cout << " > Allocating Device Memory for Image Buffers..."
              << std::endl;
  const size_t img_size = image.total() * image.elemSize();
  checkCudaError(cudaMalloc(&d_pointers.d_image, img_size),
                 "cudaMalloc failed for d_image");
  checkCudaError(cudaMalloc(&d_pointers.d_image_out, img_size),
                 "cudaMalloc failed for d_image_out");
  checkCudaError(cudaMalloc(&d_pointers.d_flow, img_size),
                 "cudaMalloc failed for d_flow");

  // Image is already unstacked and padded, transfer directly
  checkCudaError(cudaMemcpy(d_pointers.d_image, image.data, img_size,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy failed to device");
}

void transfer_back_and_cleanup(D_pointers &d_pointers, cv::Mat &image) {
  const size_t img_size = image.total() * image.elemSize();

  // Image will be unstacked and stacked on CPU, transfer directly
  checkCudaError(cudaMemcpy(image.data, d_pointers.d_image, img_size,
                            cudaMemcpyDeviceToHost),
                 "Error cudaMemcpy for image");

  cudaFree(d_pointers.d_permutation_vector);
  cudaFree(d_pointers.d_permutation_vector_inverse);
  cudaFree(d_pointers.d_permutation_blocks);
  cudaFree(d_pointers.d_permutation_blocks_inverse);
  cudaFree(d_pointers.d_seeds);
  cudaFree(d_pointers.d_r_params);
  cudaFree(d_pointers.d_flow);
  cudaFree(d_pointers.d_image);
  cudaFree(d_pointers.d_image_out);
  cudaFree(d_pointers.d_automata_state);
  cudaFree(d_pointers.d_image_automata_state);
  cudaFree(d_pointers.d_chaotic_values_for_permutation);
  cudaFree(d_pointers.d_permutation_blocks_inital);
  cudaFree(d_pointers.d_permutation_blocks_inverse_initial);

}

// =================================================================================
//                                  MAIN ORCHESTRATOR
// =================================================================================

__host__ void encrypt_image(cv::Mat &image,
                            std::vector<std::vector<unsigned char>> &password,
                            const Image_dimensions &img_dimensions,
                            const EncryptionParams &params, bool verbose,
                            bool encrypt) {
  D_pointers d_pointers;

  if (verbose)
    print_encryption_report(image, img_dimensions, params, encrypt);

  allocate_and_transfer_image(d_pointers, image, params, verbose);

  setup_permutations(d_pointers, password, img_dimensions, params, verbose);

  // For flow and block permutations (and extra seeds)
  convert_bits_to_real(password[1], &d_pointers.d_seeds);

  // Per-seed chaotic r parameters derived from key
  convert_bits_to_r_params(password[2], &d_pointers.d_r_params);

  if (verbose)
    std::cout << " > Starting GPU Execution..." << std::endl;

  if (encrypt) {
    encryption_process(d_pointers, img_dimensions, params.block_size, params,
                       verbose);
  } else {
    unencryption_process(d_pointers, img_dimensions, params.block_size, params,
                         verbose);
  }

  if (verbose)
    std::cout << " > GPU Execution Completed." << std::endl;

  transfer_back_and_cleanup(d_pointers, image);
}

// =================================================================================
//                       ENCRYPTION & DECRYPTION PROCESSES
// =================================================================================

/**
 * @brief Main encryption process implementing confusion-diffusion rounds.
 *
 * Encryption Flow:
 *  1. Initial Confusion: Permute image (rows/cols/blocks)
 *  2. Rounds Loop:
 *     a) Generate chaotic keystream
 *     b) Permute keystream
 *     c) XOR image with permuted keystream (diffusion)
 *  3. Final Confusion: Permute image again
 */
void encryption_process(D_pointers &d_pointers, Image_dimensions img_dimensions,
                        size_t block_size, const EncryptionParams &params,
                        bool verbose) {

  if (verbose)
    std::cout << " > Starting Encryption (" << params.rounds << " rounds)..."
              << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  if (verbose)
    std::cout << " > Generating Initial Permutations..." << std::endl;

  // Generate initial stream (transition)
  auto transition_start = std::chrono::high_resolution_clock::now();
  generate_flow_stream_parallel(d_pointers, img_dimensions, params);
  auto transition_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> transition_time =
      transition_end - transition_start;
  if (verbose)
    std::cout << "\tInitial stream generated in "
              << transition_time.count() * 1000.0f << " ms" << std::endl;

  transition_start = std::chrono::high_resolution_clock::now();
  // Generate block permutations
  generate_permutation_block(d_pointers, img_dimensions, params);
  transition_end = std::chrono::high_resolution_clock::now();
  transition_time = transition_end - transition_start;
  if (verbose)
    std::cout << "\tBlock permutations generated in "
              << transition_time.count() * 1000.0f << " ms" << std::endl;

  // === PHASE 1: Initial Confusion (permutation of image) ===
  if (verbose)
    std::cout << " > Performing Initial Image Permutation..." << std::endl;
  fused_permutation_xor(
      d_pointers.d_image, d_pointers.d_image_out, nullptr,
      d_pointers.d_permutation_vector, d_pointers.d_permutation_vector_inverse,
      d_pointers.d_permutation_blocks, d_pointers.d_permutation_blocks_inverse,
      img_dimensions, block_size, false, false);
  std::swap(d_pointers.d_image, d_pointers.d_image_out);

  // Inital permutation for image must be preserved for final confussion.
  std::swap(d_pointers.d_permutation_blocks, d_pointers.d_permutation_blocks_inital);
  std::swap(d_pointers.d_permutation_blocks_inverse, d_pointers.d_permutation_blocks_inverse_initial);

  // === PHASE 2: Confusion-Diffusion Rounds (permutation of keystream and
  // diffusion (XOR)) ===
  if (verbose)
    std::cout << " > Starting Confusion-Diffusion Rounds..." << std::endl;
  for (size_t round = 0; round < params.rounds; round++) {
    auto round_start = std::chrono::high_resolution_clock::now();

    auto keystream_start = std::chrono::high_resolution_clock::now();
    // Step A: Generate chaotic keystream
    generate_flow_stream_parallel(d_pointers, img_dimensions, params);
    auto keystream_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> keystream_time =
        keystream_end - keystream_start;
    if (verbose)
      std::cout << "\t\t Keystream generated in "
                << keystream_time.count() * 1000.0f << " ms" << std::endl;
    
    keystream_start = std::chrono::high_resolution_clock::now();            
    generate_permutation_block(d_pointers, img_dimensions, params);
    keystream_end = std::chrono::high_resolution_clock::now();
    keystream_time = keystream_end - keystream_start;
    if (verbose)
      std::cout << "\tBlock permutations generated in "
              << transition_time.count() * 1000.0f << " ms" << std::endl;

    keystream_start = std::chrono::high_resolution_clock::now();
    // Step B & C: Permute the keystream and apply Diffusion (XOR)
    // We do one iteration of permutation normally on d_flow,
    // and the second iteration is fused with the XOR operation on d_image.
    fused_permutation_xor(d_pointers.d_image, d_pointers.d_image_out,
                          d_pointers.d_flow, d_pointers.d_permutation_vector,
                          d_pointers.d_permutation_vector_inverse,
                          d_pointers.d_permutation_blocks,
                          d_pointers.d_permutation_blocks_inverse,
                          img_dimensions, block_size, true, false);
    std::swap(d_pointers.d_image, d_pointers.d_image_out);

    keystream_end = std::chrono::high_resolution_clock::now();
    keystream_time = keystream_end - keystream_start;
    if (verbose)
      std::cout << "\t\t Keystream permutation + Diffusion (XOR) completed in "
                << keystream_time.count() * 1000.0f << " ms" << std::endl;

    auto round_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> round_time = round_end - round_start;

    if (verbose)
      std::cout << "\tRound " << round + 1 << "/" << params.rounds
                << " complete. Time: " << round_time.count() * 1000.0f << " ms"
                << std::endl;
  }
  // Restore initial block permutations for final confusion
  std::swap(d_pointers.d_permutation_blocks, d_pointers.d_permutation_blocks_inital);
  std::swap(d_pointers.d_permutation_blocks_inverse, d_pointers.d_permutation_blocks_inverse_initial);

  // === PHASE 3: Final Confusion (the same as the initial confusion) ===

  fused_permutation_xor(
      d_pointers.d_image, d_pointers.d_image_out, nullptr,
      d_pointers.d_permutation_vector, d_pointers.d_permutation_vector_inverse,
      d_pointers.d_permutation_blocks, d_pointers.d_permutation_blocks_inverse,
      img_dimensions, block_size, false, false);
  std::swap(d_pointers.d_image, d_pointers.d_image_out);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  if (verbose)
    std::cout << " > Encryption Complete: " << time.count() * 1000.0f << " ms"
              << std::endl;
}

/**
 * @brief Main decryption process - reverses the encryption operations.
 *
 * Decryption Flow (reverse of encryption):
 *  1. Reverse Final Confusion: Inverse permutation
 *  2. Rounds Loop (same count as encryption):
 *     a) Regenerate same chaotic keystream
 *     b) Permute keystream (same way)
 *     c) XOR to reverse diffusion
 *  3. Reverse Initial Confusion: Inverse permutation
 */
void unencryption_process(D_pointers &d_pointers,
                          Image_dimensions img_dimensions, size_t block_size,
                          const EncryptionParams &params, bool verbose) {
  if (verbose)
    std::cout << " > Starting Decryption (" << params.rounds << " rounds)..."
              << std::endl;

  // Generate stream and block permutations (same as encryption)
  generate_flow_stream_parallel(d_pointers, img_dimensions, params);

  generate_permutation_block(d_pointers, img_dimensions, params);

  // === PHASE 1: Reverse Final Confusion ===
  fused_permutation_xor(
      d_pointers.d_image, d_pointers.d_image_out, nullptr,
      d_pointers.d_permutation_vector, d_pointers.d_permutation_vector_inverse,
      d_pointers.d_permutation_blocks, d_pointers.d_permutation_blocks_inverse,
      img_dimensions, block_size, false, true);
  std::swap(d_pointers.d_image, d_pointers.d_image_out);

  std::swap(d_pointers.d_permutation_blocks, d_pointers.d_permutation_blocks_inital);
  std::swap(d_pointers.d_permutation_blocks_inverse, d_pointers.d_permutation_blocks_inverse_initial);

  // === PHASE 2: Reverse Diffusion-Confusion Rounds ===
  for (size_t round = 0; round < params.rounds; round++) {
    // Step A: Regenerate exact same chaotic keystream
    generate_flow_stream_parallel(d_pointers, img_dimensions, params);

    generate_permutation_block(d_pointers, img_dimensions, params);

    // Step B & C: Permute keystream and apply Fusion (XOR)
    fused_permutation_xor(d_pointers.d_image, d_pointers.d_image_out,
                          d_pointers.d_flow, d_pointers.d_permutation_vector,
                          d_pointers.d_permutation_vector_inverse,
                          d_pointers.d_permutation_blocks,
                          d_pointers.d_permutation_blocks_inverse,
                          img_dimensions, block_size, true, false);
    std::swap(d_pointers.d_image, d_pointers.d_image_out);
  }

  // === PHASE 3: Reverse Initial Confusion ===

  std::swap(d_pointers.d_permutation_blocks, d_pointers.d_permutation_blocks_inital);
  std::swap(d_pointers.d_permutation_blocks_inverse, d_pointers.d_permutation_blocks_inverse_initial);

  fused_permutation_xor(
      d_pointers.d_image, d_pointers.d_image_out, nullptr,
      d_pointers.d_permutation_vector, d_pointers.d_permutation_vector_inverse,
      d_pointers.d_permutation_blocks, d_pointers.d_permutation_blocks_inverse,
      img_dimensions, block_size, false, true);
  std::swap(d_pointers.d_image, d_pointers.d_image_out);
}
