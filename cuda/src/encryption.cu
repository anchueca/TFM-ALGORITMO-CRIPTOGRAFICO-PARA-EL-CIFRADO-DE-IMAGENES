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
      static_cast<size_t>(image.rows)
  };

  D_pointers d_pointers;

  // Calculate grid block distribution
  const size_t num_blocks_per_row =
      img_dimensions.rows / params.block_size +
      (img_dimensions.rows % params.block_size != 0);
  const size_t num_blocks_per_col =
      img_dimensions.cols / params.block_size +
      (img_dimensions.cols % params.block_size != 0);
  const size_t num_blocks = num_blocks_per_row * num_blocks_per_col;
  const size_t num_blocks_permutations = 1;
  const size_t block_data_length = params.block_size * params.block_size;

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
  const std::vector<std::vector<unsigned char>> password_segments =
      calculate_password(password, num_blocks_permutations,
                         params.precision_level, img_dimensions, verbose);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;

  if (verbose)
    std::cout << " > Password hashing & expansion: " << time.count() << " s"
              << std::endl;

  // --- 4. PERMUTATION GENERATION (GPU) ---
  if (verbose)
    std::cout << " > Generating Permutations..." << std::endl;

  // A. Columns
  ElementalCelularAutomata automata(password_segments[1],
                                    img_dimensions.cols * 2 * 8, 30);
  const std::vector<ElementalCelularAutomata *> cols_automata = {&automata};
  d_pointers.d_permutation_cols = generate_automata_permutations(
      cols_automata, params.automata_steps, img_dimensions.cols, verbose);

  // B. Rows
  ElementalCelularAutomata automata1(password_segments[0],
                                     img_dimensions.rows * 2 * 8, 30);
  const std::vector<ElementalCelularAutomata *> rows_automata = {&automata1};
  d_pointers.d_permutation_rows = generate_automata_permutations(
      rows_automata, params.automata_steps, img_dimensions.rows, verbose);

  // C. Blocks (Chaotic Map)
  d_pointers.d_permutation_blocks = generate_flow_permutations(
      password_segments[2], block_data_length, num_blocks_permutations,
      params.transition_length, params.chaos_parameter);

  // --- 5. INVERSE PERMUTATIONS ---
  if (verbose)
    std::cout << " > Calculating Inverse Permutations..." << std::endl;
  inverse_permutations(d_pointers.d_permutation_cols,
                       &d_pointers.d_permutation_cols_inverse,
                       img_dimensions.cols, 1);
  inverse_permutations(d_pointers.d_permutation_rows,
                       &d_pointers.d_permutation_rows_inverse,
                       img_dimensions.rows, 1);
  inverse_permutations(d_pointers.d_permutation_blocks,
                       &d_pointers.d_permutation_blocks_inverse,
                       block_data_length, num_blocks_permutations);

  // --- 6. MEMORY ALLOCATION & DATA TRANSFER ---
  
  // Total size is same as original interleaved image.
  const size_t img_size = image.total() * image.elemSize();

  cudaMalloc(&d_pointers.d_image, img_size);
  cudaMalloc(&d_pointers.d_image_out, img_size);
  cudaMalloc(&d_pointers.d_flow, img_size);

  if (is_color) {
      // 1. Upload Interleaved data to d_image_out (as temp buffer)
      cudaMemcpy(d_pointers.d_image_out, image.data, img_size, cudaMemcpyHostToDevice);
      
      // 2. GPU Unstack: d_image_out (Interleaved) -> d_image (Planar)
      unstack_channels_gpu(d_pointers.d_image_out, d_pointers.d_image, image.cols, image.rows);
  } else {
      // Grayscale: Direct copy
      cudaMemcpy(d_pointers.d_image, image.data, img_size, cudaMemcpyHostToDevice);
  }

  convert_bits_to_real(password_segments[3], &d_pointers.d_seeds);

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
      
      stack_channels_gpu(d_pointers.d_image, d_pointers.d_image_out, image.cols, image.rows);
      
      // Download from d_image_out
      cudaMemcpy(image.data, d_pointers.d_image_out, img_size, cudaMemcpyDeviceToHost);
  } else {
      cudaMemcpy(image.data, d_pointers.d_image, img_size, cudaMemcpyDeviceToHost);
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
//                            PROCESS FLOW LOGIC
// =================================================================================

void encryption_process(D_pointers &d_pointers, Image_dimensions img_dimensions,
                        size_t block_size, const EncryptionParams &params,
                        bool verbose) {

  if (verbose)
    std::cout << " > Starting Encryption Loop (" << params.rounds << " rounds)"
              << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // 1. Initial Confusion
  image_permutation_encryption_process(d_pointers, img_dimensions, block_size);

  // 2. Diffusion + Confusion Rounds
  for (size_t i = 0; i < params.rounds; i++) {
    // A. Generate Chaotic Stream
    generate_flow_stream_parallel(d_pointers, img_dimensions, params.chaos_parameter,
                         params.transition_length);

    // B. Permute the Stream (not the image)
    permutation_encryption_process(d_pointers, img_dimensions, block_size);

    // C. Diffusion (Image XOR Stream)
    flow_encrypt(d_pointers, img_dimensions);

    if (verbose)
      std::cout << "\tRound " << i + 1 << "/" << params.rounds << " complete."
                << std::endl;
  }

  // 3. Final Confusion
  image_permutation_encryption_process(d_pointers, img_dimensions, block_size);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  if (verbose)
    std::cout << " > Total Loop Time: " << time.count() << " s" << std::endl;
}

void unencryption_process(D_pointers &d_pointers,
                          Image_dimensions img_dimensions, size_t block_size,
                          const EncryptionParams &params, bool verbose) {
  if (verbose)
    std::cout << " > Starting Decryption Loop..." << std::endl;

  // 1. Reverse Final Confusion
  image_permutation_unencryption_process(d_pointers, img_dimensions,
                                         block_size);

  // 2. Reverse Rounds
  for (size_t i = 0; i < params.rounds; i++) {
    // Regenerate exact same flow
    generate_flow_stream_parallel(d_pointers, img_dimensions, params.chaos_parameter,
                         params.transition_length);
    permutation_encryption_process(d_pointers, img_dimensions, block_size);

    // Inverse XOR (Identical to forward XOR)
    flow_encrypt(d_pointers, img_dimensions);
  }

  // 3. Reverse Initial Confusion
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

cv::Mat unstack_channels(const cv::Mat &image, bool verbose) {
  cv::Mat processed_image;
  if (image.channels() == 3) {
    if (verbose)
      std::cout << "[INFO] 3-Channel image detected. Unstacking..."
                << std::endl;
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::hconcat(channels, processed_image);
  } else {
    processed_image = image.clone();
  }
  return processed_image;
}

void stack_channels(cv::Mat &image, const cv::Mat &processed_image,
                    bool is_color, bool verbose) {
  if (is_color) {
    if (verbose)
      std::cout << "[INFO] Restacking channels back to RGB..." << std::endl;

    int w = processed_image.cols / 3;
    int h = processed_image.rows;

    cv::Mat b = processed_image(cv::Rect(0, 0, w, h));
    cv::Mat g = processed_image(cv::Rect(w, 0, w, h));
    cv::Mat r = processed_image(cv::Rect(2 * w, 0, w, h));

    std::vector<cv::Mat> channels = {b, g, r};
    cv::merge(channels, image);
  } else {
    image = processed_image;
  }
}