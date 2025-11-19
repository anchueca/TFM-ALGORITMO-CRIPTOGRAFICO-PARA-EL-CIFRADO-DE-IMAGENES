#include "../include/encryption.cuh"

// Top-level encryption orchestration (implementation).
// See `include/encryption.cuh` for the API and parameter descriptions.
__host__ void encrypt_image(cv::Mat image, const std::string &password,
                            const EncryptionParams &params, bool verbose,
                            bool encrypt) {

  // For now we assume the image dimensions are multiples of block_size
  const Image_dimnesions img_dimensions = {static_cast<size_t>(image.cols),
                                           static_cast<size_t>(image.rows)};

  D_pointers d_pointers;

  const size_t num_blocks_per_row =
      img_dimensions.rows / params.block_size +
      (img_dimensions.rows % params.block_size != 0);
  const size_t num_blocks_per_col =
      img_dimensions.cols / params.block_size +
      (img_dimensions.cols % params.block_size != 0);
  const size_t num_blocks = num_blocks_per_row * num_blocks_per_col;
  const size_t num_blocks_permutations = 1; //num_blocks // Case multiple permutation blocks
  const size_t block_data_length = params.block_size * params.block_size;

  auto start = std::chrono::high_resolution_clock::now();
  const std::vector<std::vector<unsigned char>> password_segments =
      calculate_password(password, num_blocks_permutations, params.precision_level, img_dimensions);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  if (verbose)
    std::cout << "Password generation time: " << time.count() << " s"
              << std::endl;

  if (verbose) {
    std::cout << "=== Encryption parameters ===" << std::endl;
    std::cout << "\tPrecision level: " << params.precision_level << std::endl;
    std::cout << "\tAutomata steps: " << params.automata_steps << std::endl;
    std::cout << "\tTransition length: " << params.transition_length
              << std::endl;
    std::cout << "\tBlock size: " << params.block_size << std::endl;
    std::cout << "\tNum blocks per row: " << num_blocks_per_row << std::endl;
    std::cout << "\tNum blocks per col: " << num_blocks_per_col << std::endl;
    std::cout << "\tNum blocks: " << num_blocks << std::endl;
    std::cout << "\tBlock data length: " << block_data_length << std::endl;
    std::cout << "===========================" << std::endl << std::endl;
  }

  // Automatas
  if (verbose)
    std::cout << "Generating row and column permutations using Elemental "
                 "Cellular Automata..."
              << std::endl;
  ElementalCelularAutomata automata(
      password_segments[1], img_dimensions.cols * params.precision_level * 8,
      30);
  const std::vector<ElementalCelularAutomata *> cols_automata = {&automata};

  start = std::chrono::high_resolution_clock::now();
  d_pointers.d_permutation_cols = generate_automata_permutations(
      cols_automata, params.automata_steps, img_dimensions.cols);
  end = std::chrono::high_resolution_clock::now();
  time = end - start;
  if (verbose)
    std::cout << "\t\tgenerate_automata_permutations (cols) time: "
              << time.count() << " s" << std::endl;

  ElementalCelularAutomata automata1(
      password_segments[0], img_dimensions.rows * params.precision_level * 8,
      30);
  const std::vector<ElementalCelularAutomata *> rows_automata = {&automata1};

  start = std::chrono::high_resolution_clock::now();
  d_pointers.d_permutation_rows = generate_automata_permutations(
      rows_automata, params.automata_steps, img_dimensions.rows);
  end = std::chrono::high_resolution_clock::now();
  time = end - start;
  if (verbose)
    std::cout << "\t\tgenerate_automata_permutations (rows) time: "
              << time.count() << " s" << std::endl;

  // Generate permutations
  if (verbose)
    std::cout << ("Generating block permutations using chaotic function...")
              << std::endl;

  start = std::chrono::high_resolution_clock::now();
  d_pointers.d_permutation_blocks =
      generate_flow_permutations(password_segments[2], block_data_length,
                                 num_blocks_permutations, params.transition_length);
  end = std::chrono::high_resolution_clock::now();
  time = end - start;
  if (verbose)
    std::cout << "\t\tgenerate_flow_permutations (blocks) time: "
              << time.count() << " s" << std::endl;

  const size_t img_size = image.total() * image.elemSize();

  // Calulate inverse permutations
  if (verbose)
    std::cout << "Calculating inverse permutations..." << std::endl;

    start = std::chrono::high_resolution_clock::now();
    inverse_permutations(d_pointers.d_permutation_cols,&d_pointers.d_permutation_cols_inverse, img_dimensions.cols,
                          1);
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    if (verbose)
      std::cout << "\tinverse cols time: " << time.count() << " s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    inverse_permutations(d_pointers.d_permutation_rows,&d_pointers.d_permutation_rows_inverse, img_dimensions.rows,
                          1);
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    if (verbose)
      std::cout << "\tinverse rows time: " << time.count() << " s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    inverse_permutations(d_pointers.d_permutation_blocks,&d_pointers.d_permutation_blocks_inverse, block_data_length,
                          num_blocks_permutations);
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    if (verbose)
      std::cout << "\tinverse blocks time: " << time.count() << " s"
                << std::endl;

  cudaMalloc(&d_pointers.d_image, img_size);
  cudaMalloc(&d_pointers.d_image_out, img_size);
  cudaMalloc(&d_pointers.d_flow, img_size);
  cudaMalloc(&d_pointers.d_seeds, password_segments[3].size() * sizeof(unsigned char));

  cudaMemcpy(d_pointers.d_image, image.data, img_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pointers.d_seeds, password_segments[3].data(), password_segments[3].size() * sizeof(unsigned char), cudaMemcpyHostToDevice);

  if (encrypt) {
    encryption_process(d_pointers, img_dimensions,
                       params.block_size, params.rounds, verbose);
  } else {
    unencryption_process(d_pointers, img_dimensions,
                         params.block_size, params.rounds);
  }
  if (verbose)
    std::cout << "Completed" << std::endl;

  cudaMemcpy(image.data, d_pointers.d_image, img_size, cudaMemcpyDeviceToHost);

  cudaFree(d_pointers.d_permutation_cols);
  cudaFree(d_pointers.d_permutation_rows);
  cudaFree(d_pointers.d_permutation_blocks);

  cudaFree(d_pointers.d_seeds);
  cudaFree(d_pointers.d_flow);
  cudaFree(d_pointers.d_image);
  cudaFree(d_pointers.d_image_out);
}

void encryption_process(D_pointers &d_pointers, Image_dimnesions img_dimensions,
                        size_t block_size, size_t rounds, bool verbose) {

  if (verbose)
    std::cout << "Starting encryption with " << rounds << " rounds."
              << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  image_permutation_encryption_process(d_pointers, img_dimensions, block_size);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  std::cout << "\tInitial image_permutation_encryption_proces time: "
            << time.count() << " s" << std::endl;

  // Rounds
  for (size_t i = 0; i < rounds; i++) {
    start = std::chrono::high_resolution_clock::now();
    auto start1 = std::chrono::high_resolution_clock::now();
    generate_flow_stream(d_pointers, img_dimensions,3.999);
    permutation_encryption_process(d_pointers, img_dimensions, block_size);// For flow
    flow_encrypt(d_pointers,img_dimensions);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time1 = end1 - start1;
    std::cout << "\t\t\t\tflow_encrypt(" << i << ")"
              << " time: " << time1.count() << " s" << std::endl;

    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    if (verbose)
      std::cout << "\t\t\tround(" << i << ")" << " time: " << time.count() << " s"
              << std::endl;
  }

  start = std::chrono::high_resolution_clock::now();

  image_permutation_encryption_process(d_pointers, img_dimensions, block_size);

  end = std::chrono::high_resolution_clock::now();
  time = end - start;
  if (verbose)
    std::cout << "\tFinal image_permutation_encryption_process time: "
            << time.count() << " s" << std::endl;
}

void unencryption_process(D_pointers &d_pointers,
                          Image_dimnesions img_dimensions,
                          size_t block_size, size_t rounds) {

  std::cout << "Starting decryption with " << rounds << " rounds." << std::endl;

  image_permutation_unencryption_process(d_pointers, img_dimensions,block_size);

  // Rounds
  for (size_t i = 0; i < rounds; i++) {
    generate_flow_stream(d_pointers, img_dimensions,3.999);
    permutation_encryption_process(d_pointers, img_dimensions, block_size);
    flow_encrypt(d_pointers,img_dimensions);
  }

  image_permutation_unencryption_process(d_pointers, img_dimensions,block_size);
}

void image_permutation_encryption_process(D_pointers &d_pointers,
                                          Image_dimnesions img_dimensions,
                                          size_t block_size) {
  unsigned char *temp = nullptr;
  for (size_t j = 0; j < 2; j++) {
    // Rows and columns
    rows_and_columns_permutation(d_pointers.d_image, d_pointers.d_image_out,
                                 d_pointers.d_permutation_rows,
                                 d_pointers.d_permutation_cols, img_dimensions,
                                 false);
    // Block
    block_phase_permutation_simple(d_pointers.d_image, d_pointers.d_image_out,
                            d_pointers.d_permutation_blocks,d_pointers.d_permutation_blocks_inverse, img_dimensions,
                            block_size);
    temp = d_pointers.d_image;
    d_pointers.d_image = d_pointers.d_image_out;
    d_pointers.d_image_out = temp;
  }
}

void permutation_encryption_process(D_pointers &d_pointers,
                                    Image_dimnesions img_dimensions,
                                    size_t block_size) {
  unsigned char *temp = nullptr;
  for (size_t j = 0; j < 2; j++) {
    // Rows and columns
    rows_and_columns_permutation(d_pointers.d_flow, d_pointers.d_image_out,
                                 d_pointers.d_permutation_rows,
                                 d_pointers.d_permutation_cols, img_dimensions,
                                 false);
    // Block
    block_phase_permutation_simple(d_pointers.d_flow, d_pointers.d_image_out,
                            d_pointers.d_permutation_blocks,d_pointers.d_permutation_blocks_inverse, img_dimensions,
                            block_size);
    temp = d_pointers.d_flow;
    d_pointers.d_flow = d_pointers.d_image_out;
    d_pointers.d_image_out = temp;
  }
}

void image_permutation_unencryption_process(D_pointers &d_pointers,
                                            Image_dimnesions img_dimensions,
                                            size_t block_size) {
  for (size_t j = 0; j < 2; j++) {
    unsigned char *temp = nullptr;
    // Block
    block_phase_permutation_simple(d_pointers.d_image, d_pointers.d_image_out,
                            d_pointers.d_permutation_blocks_inverse,d_pointers.d_permutation_blocks, img_dimensions,
                            block_size);

    temp = d_pointers.d_image;
    d_pointers.d_image = d_pointers.d_image_out;
    d_pointers.d_image_out = temp;

    // Rows and columns
    rows_and_columns_permutation(d_pointers.d_image, d_pointers.d_image_out,
                                 d_pointers.d_permutation_rows_inverse,
                                 d_pointers.d_permutation_cols_inverse, img_dimensions,
                                 true);
  }
}