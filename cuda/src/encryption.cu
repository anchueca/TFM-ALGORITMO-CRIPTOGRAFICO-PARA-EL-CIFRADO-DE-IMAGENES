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
  std::cout << "  " << std::left << std::setw(25)
            << "Chaotic Parameter (r):" << params.chaos_parameter << std::endl;
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
  ElementalCelularAutomata cols_automata(password[0],
                                         img_dimensions.cols * 2 * 8, 30);
  d_pointers.d_permutation_cols = generate_automata_permutations(
      &cols_automata, params.automata_steps, img_dimensions.cols, verbose);

  // Copy CA state to a persistent buffer for use during flow generation
  size_t state_size = cols_automata.get_size_in_bytes();
  cudaMalloc(&d_pointers.d_automata_state, state_size);
  cudaMemcpy(d_pointers.d_automata_state, cols_automata.get_cuda_state(),
             state_size, cudaMemcpyDeviceToDevice);

  if (verbose)
    std::cout << " > Calculating Inverse Permutations..." << std::endl;
  inverse_permutations(d_pointers.d_permutation_cols,
                       &d_pointers.d_permutation_cols_inverse,
                       img_dimensions.cols, 1);
}

void allocate_and_transfer_image(D_pointers &d_pointers, cv::Mat &image,
                                 const EncryptionParams &params) {
  const size_t img_size = image.total() * image.elemSize();
  cudaMalloc(&d_pointers.d_image, img_size);
  cudaMalloc(&d_pointers.d_image_out, img_size);

  cudaMalloc(&d_pointers.d_flow, img_size);

  // Image is already unstacked and padded, transfer directly
  cudaMemcpy(d_pointers.d_image, image.data, img_size, cudaMemcpyHostToDevice);
}

void transfer_back_and_cleanup(D_pointers &d_pointers, cv::Mat &image) {
  const size_t img_size = image.total() * image.elemSize();

  // Image will be unstacked and stacked on CPU, transfer directly
  cudaMemcpy(image.data, d_pointers.d_image, img_size, cudaMemcpyDeviceToHost);

  cudaFree(d_pointers.d_permutation_cols);
  cudaFree(d_pointers.d_permutation_blocks);
  cudaFree(d_pointers.d_seeds);
  cudaFree(d_pointers.d_flow);
  cudaFree(d_pointers.d_image);
  cudaFree(d_pointers.d_image_out);
  cudaFree(d_pointers.d_automata_state);
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

  setup_permutations(d_pointers, password, img_dimensions, params, verbose);

  allocate_and_transfer_image(d_pointers, image, params);

  // For flow and block permutations (and extra seeds)
  convert_bits_to_real(password[1], &d_pointers.d_seeds);

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

  // Generate initial stream (transition)
  generate_flow_stream_parallel(d_pointers, img_dimensions, params);
  // Generate block permutations
  generate_permutation_block(d_pointers, img_dimensions, params);

  // === PHASE 1: Initial Confusion (permutation of image) ===
  image_permutation_encryption_process(d_pointers, img_dimensions, block_size);

  // === PHASE 2: Confusion-Diffusion Rounds (permutation of keystream and
  // diffusion (XOR)) ===
  for (size_t round = 0; round < params.rounds; round++) {
    auto round_start = std::chrono::high_resolution_clock::now();

    // Step A: Generate chaotic keystream
    generate_flow_stream_parallel(d_pointers, img_dimensions, params);

    // Step B: Permute the keystream
    permutation_encryption_process(d_pointers, img_dimensions, block_size);

    // Step C: Diffusion - XOR image with permuted keystream
    flow_encrypt(d_pointers, img_dimensions);

    auto round_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> round_time = round_end - round_start;

    if (verbose)
      std::cout << "\tRound " << round + 1 << "/" << params.rounds
                << " complete. Time: " << round_time.count() * 1000.0f << " ms"
                << std::endl;
  }

  // === PHASE 3: Final Confusion (the same as the initial confusion) ===

  image_permutation_encryption_process(d_pointers, img_dimensions, block_size);

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
  image_permutation_unencryption_process(d_pointers, img_dimensions,
                                         block_size);

  // === PHASE 2: Reverse Diffusion-Confusion Rounds ===
  for (size_t round = 0; round < params.rounds; round++) {
    // Step A: Regenerate exact same chaotic keystream
    generate_flow_stream_parallel(d_pointers, img_dimensions, params);

    // Step B: Permute keystream (same as encryption)
    permutation_encryption_process(d_pointers, img_dimensions, block_size);

    // Step C: Reverse diffusion via XOR (XOR is its own inverse)
    flow_encrypt(d_pointers, img_dimensions);
  }

  // === PHASE 3: Reverse Initial Confusion ===
  image_permutation_unencryption_process(d_pointers, img_dimensions,
                                         block_size);
}

// =================================================================================
//                            PERMUTATION STAGES
// =================================================================================

void image_permutation_encryption_process(D_pointers &d_pointers,
                                          Image_dimensions img_dimensions,
                                          size_t block_size) {
  for (size_t j = 0; j < 2; j++) {
    // 1. Rows and Columns
    rows_and_columns_permutation(d_pointers.d_image, d_pointers.d_image_out,
                                 d_pointers.d_permutation_cols,
                                 d_pointers.d_permutation_cols_inverse,
                                 img_dimensions, false);
    // 2. Blocks
    // Input: d_image (new data) -> Output: d_image_out (free buffer)
    block_phase_permutation(d_pointers.d_image, d_pointers.d_image_out,
                            d_pointers.d_permutation_blocks,
                            d_pointers.d_permutation_blocks_inverse,
                            img_dimensions, block_size);

    std::swap(d_pointers.d_image, d_pointers.d_image_out);
  }
}

void image_permutation_unencryption_process(D_pointers &d_pointers,
                                            Image_dimensions img_dimensions,
                                            size_t block_size) {
  for (size_t j = 0; j < 2; j++) {
    // 1. Inverse Blocks
    block_phase_permutation(d_pointers.d_image, d_pointers.d_image_out,
                            d_pointers.d_permutation_blocks_inverse,
                            d_pointers.d_permutation_blocks, img_dimensions,
                            block_size);
    std::swap(d_pointers.d_image, d_pointers.d_image_out);

    // 2. Inverse Rows and Columns
    rows_and_columns_permutation(d_pointers.d_image, d_pointers.d_image_out,
                                 d_pointers.d_permutation_cols_inverse,
                                 d_pointers.d_permutation_cols, img_dimensions,
                                 true);
  }
}

void permutation_encryption_process(D_pointers &d_pointers,
                                    Image_dimensions img_dimensions,
                                    size_t block_size) {
  // Operation on d_flow, using d_image_out as temp buffer
  for (size_t j = 0; j < 2; j++) {
    // Rows and Columns on Flow
    rows_and_columns_permutation(d_pointers.d_flow, d_pointers.d_image_out,
                                 d_pointers.d_permutation_cols,
                                 d_pointers.d_permutation_cols_inverse,
                                 img_dimensions, false);
    // Blocks on Flow
    block_phase_permutation(d_pointers.d_flow, d_pointers.d_image_out,
                            d_pointers.d_permutation_blocks,
                            d_pointers.d_permutation_blocks_inverse,
                            img_dimensions, block_size);
    std::swap(d_pointers.d_flow, d_pointers.d_image_out);
  }
}
