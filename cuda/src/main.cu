#include "../include/cli_config.cuh"
#include "../include/encryption.cuh"

#include <chrono>
#include <cstring>
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
  AppConfig config;
  if (!parse_arguments(argc, argv, config)) {
    print_usage(argv[0]);
    return -1;
  }

  cv::Mat image = load_image(config.input_image_path);
  if (image.empty()) {
    cerr << "[ERROR] Image data is empty or corrupted." << endl;
    return -1;
  }

  print_initial_report(config, image);

  cudaFree(0);
  warmup_gpu();

  // Store original channel info
  int original_channels = image.channels();

  // === PREPROCESSING: Unstack and (optionally) Pad ===
  if (config.verbose)
    std::cout << " > Preprocessing (Unstack & Pad)..." << std::endl;

  if (config.verbose)
    std::cout << " [DEBUG] Loaded image: " << image.cols << "x" << image.rows
              << " channels=" << image.channels() << std::endl;

  cv::Mat processed_image = unstack_channels(image, config.verbose);

  if (config.verbose)
    std::cout << " [DEBUG] After unstack: " << processed_image.cols << "x"
              << processed_image.rows
              << " channels=" << processed_image.channels() << std::endl;

  // Only pad during ENCRYPTION (encrypted images are already padded)
  if (config.encrypt) {
    processed_image = padImageToSquare(
        processed_image, config.params.block_size, original_channels);
  }

  // Calculate dimensions AFTER padding (or after unstack for decryption)
  const Image_dimensions img_dimensions = {
      static_cast<size_t>(processed_image.cols),
      static_cast<size_t>(processed_image.rows)};

  if (config.verbose)
    std::cout << " [INFO] Processing dimensions: " << img_dimensions.cols
              << " x " << img_dimensions.rows << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  if (config.verbose)
    std::cout << " > Password hashing & expansion: ";

  std::vector<std::vector<unsigned char>> password_segments =
      calculate_password(config.password, img_dimensions, config.verbose,
                         config.use_raw_key);

  auto end = std::chrono::high_resolution_clock::now();
  if (config.verbose)
    std::cout << std::chrono::duration<double>(end - start).count() * 1000.0f
              << " ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();

  // Ophuscated phase
  try {
    if (config.encrypt) {
      config.params.image_hash = calculate_image_hash(processed_image, 2);
      if (config.verbose)
        std::cerr << " [INFO] Calculated Image Hash: "
                  << config.params.image_hash << std::endl;
    } else {
      // Recovery the ophuscated image hash
      config.params.image_hash =
          extract_message_caos(processed_image, password_segments[2],
                               config.input_image_path, config.exif_hex);
      if (config.verbose)
        std::cerr << " [INFO] Recovered Image Hash: "
                  << config.params.image_hash << std::endl;
    }
    // config.params.image_hash = 43243; // REMOVE
  } catch (const std::exception &e) {
    cerr << "\n[FATAL ERROR] During image hash process: " << e.what() << endl;
    return -1;
  }
  end = std::chrono::high_resolution_clock::now();

  start = std::chrono::high_resolution_clock::now();
  try {
    encrypt_image(processed_image, password_segments, img_dimensions,
                  config.params, config.verbose, config.encrypt);
  } catch (const std::exception &e) {
    cerr << "\n[FATAL ERROR] During encryption process: " << e.what() << endl;
    return -1;
  }

  // === POSTPROCESSING: Unpad and Stack ===
  if (config.verbose)
    std::cout << " > Postprocessing (Unpad & Stack)..." << std::endl;

  if (!config.encrypt) {
    // DECRYPTION: unpad (which retrieves original_channels) then stack back to
    // original format
    if (config.verbose)
      std::cout << " [DEBUG] Before unpad: " << processed_image.cols << "x"
                << processed_image.rows
                << " channels=" << processed_image.channels() << std::endl;

    int retrieved_channels = 1;
    processed_image = unpadFromSquare(processed_image, &retrieved_channels);

    if (config.verbose)
      std::cout << " [DEBUG] After unpad: " << processed_image.cols << "x"
                << processed_image.rows
                << " channels=" << processed_image.channels()
                << " retrieved_channels=" << retrieved_channels << std::endl;

    bool is_color = (retrieved_channels == 3);
    stack_channels(image, processed_image, is_color, config.verbose);

    // No need to embed message during decryption
  } else {
    // ENCRYPTION: keep as single-channel with padding, no stacking
    // The encrypted image is saved as single-channel
    image = processed_image;

    // Embed the image hash in the image
    embed_message_caos(image, config.params.image_hash, password_segments[2],
                       config.output_arg);
  }
  end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> total_time = end - start;
  if (config.verbose) {
    std::cout << "Total Pipeline Time: " << total_time.count() * 1000.0f
              << " ms" << std::endl;
  }
  std::cerr << "EXEC_TIME:" << total_time.count() << std::endl;

  return handle_output(config.output_mode, config.output_arg, image,
                       config.verbose)
             ? 0
             : -1;
}