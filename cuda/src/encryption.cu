#include "../include/encryption.cuh"

// =================================================================================
//                                  MAIN ORCHESTRATOR
// =================================================================================

__host__ void encrypt_image(cv::Mat &image, const std::string &password,
                            const EncryptionParams &params, bool verbose,
                            bool encrypt) {

  // --- 1. PRE-PROCESSING (GPU Side Optimization) ---
  bool is_color = (image.channels() == 3);

  // Calculate dimensions locally without CPU unstacking
  // If color, the "processed" width is cols * 3.
  const Image_dimensions img_dimensions = {
      static_cast<size_t>(is_color ? image.cols * 3 : image.cols),
      static_cast<size_t>(image.rows)};

  D_pointers d_pointers;

  // Calculate grid block distribution
  const size_t num_blocks_per_row =
      img_dimensions.rows / params.block_size +
      (img_dimensions.rows % params.block_size != 0);
  const size_t num_blocks_per_col =
      img_dimensions.cols / params.block_size +
      (img_dimensions.cols % params.block_size != 0);
  const size_t num_blocks = num_blocks_per_row * num_blocks_per_col;

  // --- 2. VERBOSE REPORTING ---
  if (verbose) {
    std::cout
        << "\n============================================================"
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
              << (is_color ? "Color (RGB)" : "Grayscale (1-CH)") << std::endl;
    std::cout << "  " << std::left << std::setw(25)
              << "Processing Dims:" << img_dimensions.cols << " x "
              << img_dimensions.rows << " px" << std::endl;

    std::cout << "\n [ALGORITHM SETTINGS]" << std::endl;
    std::cout << "  " << std::left << std::setw(25)
              << "Rounds:" << params.rounds << std::endl;
    std::cout << "  " << std::left << std::setw(25)
              << "Block Size:" << params.block_size << " px" << std::endl;
    std::cout << "  " << std::left << std::setw(25)
              << "Automata Steps:" << params.automata_steps << std::endl;
    std::cout << "  " << std::left << std::setw(25)
              << "Transition Length:" << params.transition_length << std::endl;

    std::cout << "\n [GRID ARCHITECTURE]" << std::endl;
    std::cout << "  " << std::left << std::setw(25)
              << "Grid Layout:" << num_blocks_per_col << " (cols) x "
              << num_blocks_per_row << " (rows)" << std::endl;
    std::cout << "  " << std::left << std::setw(25)
              << "Total Blocks:" << num_blocks << std::endl;
    std::cout
        << "============================================================\n"
        << std::endl;
  }

  // --- 3. KEY GENERATION (Host heavy) ---

  auto start = std::chrono::high_resolution_clock::now();

  // --- 2. PASSWORD PROCESSING ---
  if (verbose)
    std::cout << " > Password hashing & expansion: ";

  std::vector<std::vector<unsigned char>> password_segments =
      calculate_password(password, params.num_blocks_permutations,
                         img_dimensions, verbose);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;

  if (verbose)
    std::cout << time.count() * 1000.0f << " ms" << std::endl;

  // --- 3. MEMORY ALLOCATION & DATA TRANSFER ---
  // (Moved up to grouping allocations, logic unchanged)

  // --- 4. PERMUTATION GENERATION (GPU) ---
  if (verbose)
    std::cout << " > Generating Permutations..." << std::endl;

  // A. Columns - Create automata from password segment for columns
  if (verbose)
    std::cout << "\t(Processing Cols Automata...)" << std::endl;
  ElementalCelularAutomata cols_automata(password_segments[1],
                                         img_dimensions.cols * 2 * 8, 30);
  d_pointers.d_permutation_cols = generate_automata_permutations(
      &cols_automata, params.automata_steps, img_dimensions.cols, verbose);

  // B. Rows - Create automata from password segment for rows
  if (verbose)
    std::cout << "\t(Processing Rows Automata...)" << std::endl;
  ElementalCelularAutomata rows_automata(password_segments[0],
                                         img_dimensions.rows * 2 * 8, 30);
  d_pointers.d_permutation_rows = generate_automata_permutations(
      &rows_automata, params.automata_steps, img_dimensions.rows, verbose);

  // --- 5. INVERSE PERMUTATIONS ---
  if (verbose)
    std::cout << " > Calculating Inverse Permutations..." << std::endl;
  inverse_permutations(d_pointers.d_permutation_cols,
                       &d_pointers.d_permutation_cols_inverse,
                       img_dimensions.cols, 1);
  inverse_permutations(d_pointers.d_permutation_rows,
                       &d_pointers.d_permutation_rows_inverse,
                       img_dimensions.rows, 1);

  // --- 6. MEMORY ALLOCATION & DATA TRANSFER ---

  // Total size is same as original interleaved image.
  const size_t img_size = image.total() * image.elemSize();

  cudaMalloc(&d_pointers.d_image, img_size);
  cudaMalloc(&d_pointers.d_image_out, img_size);
  cudaMalloc(&d_pointers.d_flow, img_size + params.num_blocks_permutations);

  if (is_color) {
    // 1. Upload Interleaved data to d_image_out (as temp buffer)
    cudaMemcpy(d_pointers.d_image_out, image.data, img_size,
               cudaMemcpyHostToDevice);

    // 2. GPU Unstack: d_image_out (Interleaved) -> d_image (Planar)
    unstack_channels_gpu(d_pointers.d_image_out, d_pointers.d_image, image.cols,
                         image.rows);
  } else {
    // Grayscale: Direct copy
    cudaMemcpy(d_pointers.d_image, image.data, img_size,
               cudaMemcpyHostToDevice);
  }

  convert_bits_to_real(password_segments[2], &d_pointers.d_seeds);

  // --- 7. EXECUTION ---
  if (encrypt) {
    encryption_process(d_pointers, img_dimensions, params.block_size, params,
                       verbose);
  } else {
    unencryption_process(d_pointers, img_dimensions, params.block_size, params,
                         verbose);
  }

  if (verbose)
    std::cout << " > GPU Execution Completed." << std::endl;

  // --- 8. POST-PROCESSING ---
  if (is_color) {
    // We need to Interleave back.
    // d_image (Planar) -> d_image_out (Interleaved)

    stack_channels_gpu(d_pointers.d_image, d_pointers.d_image_out, image.cols,
                       image.rows);

    // Download from d_image_out
    cudaMemcpy(image.data, d_pointers.d_image_out, img_size,
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(image.data, d_pointers.d_image, img_size,
               cudaMemcpyDeviceToHost);
  }

  // --- CLEANUP ---
  cudaFree(d_pointers.d_permutation_cols);
  cudaFree(d_pointers.d_permutation_rows);
  cudaFree(d_pointers.d_permutation_blocks);
  cudaFree(d_pointers.d_seeds);
  cudaFree(d_pointers.d_flow);
  cudaFree(d_pointers.d_image);
  cudaFree(d_pointers.d_image_out);
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

  // Generate initial stream and block permutations
#ifdef USE_DOUBLE_PRECISION
  generate_flow_stream_parallel<double>(d_pointers, img_dimensions, params);
#else
  generate_flow_stream_parallel<float>(d_pointers, img_dimensions, params);
#endif
  generate_permutation_block(d_pointers, img_dimensions, params);

  // === PHASE 1: Initial Confusion ===
  image_permutation_encryption_process(d_pointers, img_dimensions, block_size);

  // === PHASE 2: Confusion-Diffusion Rounds ===
  for (size_t round = 0; round < params.rounds; round++) {
    auto round_start = std::chrono::high_resolution_clock::now();

    // Step A: Generate chaotic keystream
#ifdef USE_DOUBLE_PRECISION
    generate_flow_stream_parallel<double>(d_pointers, img_dimensions, params);
#else
    generate_flow_stream_parallel<float>(d_pointers, img_dimensions, params);
#endif

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

  // === PHASE 3: Final Confusion ===
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
#ifdef USE_DOUBLE_PRECISION
  generate_flow_stream_parallel<double>(d_pointers, img_dimensions, params);
#else
  generate_flow_stream_parallel<float>(d_pointers, img_dimensions, params);
#endif
  generate_permutation_block(d_pointers, img_dimensions, params);

  // === PHASE 1: Reverse Final Confusion ===
  image_permutation_unencryption_process(d_pointers, img_dimensions,
                                         block_size);

  // === PHASE 2: Reverse Diffusion-Confusion Rounds ===
  for (size_t round = 0; round < params.rounds; round++) {
    // Step A: Regenerate exact same chaotic keystream
#ifdef USE_DOUBLE_PRECISION
    generate_flow_stream_parallel<double>(d_pointers, img_dimensions, params);
#else
    generate_flow_stream_parallel<float>(d_pointers, img_dimensions, params);
#endif

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
                                 d_pointers.d_permutation_rows,
                                 d_pointers.d_permutation_cols, img_dimensions,
                                 false);
    // 2. Blocks
    // Input: d_image (new data) -> Output: d_image_out (free buffer)
    block_phase_permutation_simple(d_pointers.d_image, d_pointers.d_image_out,
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
    block_phase_permutation_simple(d_pointers.d_image, d_pointers.d_image_out,
                                   d_pointers.d_permutation_blocks_inverse,
                                   d_pointers.d_permutation_blocks,
                                   img_dimensions, block_size);
    std::swap(d_pointers.d_image, d_pointers.d_image_out);

    // 2. Inverse Rows and Columns
    rows_and_columns_permutation(d_pointers.d_image, d_pointers.d_image_out,
                                 d_pointers.d_permutation_rows_inverse,
                                 d_pointers.d_permutation_cols_inverse,
                                 img_dimensions, true);
  }
}

void permutation_encryption_process(D_pointers &d_pointers,
                                    Image_dimensions img_dimensions,
                                    size_t block_size) {
  // Operation on d_flow, using d_image_out as temp buffer
  for (size_t j = 0; j < 2; j++) {
    // Rows and Columns on Flow
    rows_and_columns_permutation(d_pointers.d_flow, d_pointers.d_image_out,
                                 d_pointers.d_permutation_rows,
                                 d_pointers.d_permutation_cols, img_dimensions,
                                 false);
    // Blocks on Flow
    block_phase_permutation_simple(d_pointers.d_flow, d_pointers.d_image_out,
                                   d_pointers.d_permutation_blocks,
                                   d_pointers.d_permutation_blocks_inverse,
                                   img_dimensions, block_size);
    std::swap(d_pointers.d_flow, d_pointers.d_image_out);
  }
}
