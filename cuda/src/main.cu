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

  const Image_dimensions img_dimensions = {
      static_cast<size_t>(image.channels() == 3 ? image.cols * 3 : image.cols),
      static_cast<size_t>(image.rows)};

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
      config.params.image_hash = calculate_image_hash(image, 2);
      if (config.verbose)
        std::cerr << " [INFO] Calculated Image Hash: "
                  << config.params.image_hash << std::endl;
    } else {
      // Recovery the ophuscated image hash
      config.params.image_hash =
          extract_message_caos(image, password_segments[3],
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
    encrypt_image(image, password_segments, img_dimensions, config.params,
                  config.verbose, config.encrypt);
  } catch (const std::exception &e) {
    cerr << "\n[FATAL ERROR] During encryption process: " << e.what() << endl;
    return -1;
  }
  // Embed the image hash in the image
  if (config.encrypt) {
    embed_message_caos(image, config.params.image_hash, password_segments[3],
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