/**
 * @file aux.cu
 * @brief Helper implementations for image stacking, unstacking and password
 * derivation.
 */

#include "../include/aux.cuh"

// Generate SHA3-512-derived bytes (implementation; see header for API)
__host__ std::vector<unsigned char> generate_hash(const std::string &input,
                                                  size_t length) {

  // 1. Create OpenSSL context
  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  if (ctx == nullptr) {
    throw std::runtime_error("Error: Failed to create OpenSSL EVP context");
  }

  // 2. Initialize digest for SHAKE256
  // Important: SHAKE is an XOF (Extendable Output Function)
  if (EVP_DigestInit_ex(ctx, EVP_shake256(), nullptr) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("Error: Failed to initialize SHAKE256");
  }

  // 3. Absorb (Feed data)
  if (EVP_DigestUpdate(ctx, input.data(), input.size()) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("Error: Failed to update digest");
  }

  // 4. Prepare output vector of requested size (length)
  std::vector<unsigned char> output(length);

  // 5. Squeeze (Extract data)
  // For SHAKE use EVP_DigestFinalXOF, not EVP_DigestFinal_ex
  if (EVP_DigestFinalXOF(ctx, output.data(), length) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("Error: Failed to extract hash (FinalXOF)");
  }

  EVP_MD_CTX_free(ctx);

  return output;
}

// Calculate password segments from a master password (implementation)
__host__ std::vector<std::vector<unsigned char>>

calculate_password(const std::string &input, Image_dimensions img_dimensions,
                   bool verbose) {

  // Required lengths
  // Fixed to 1 as we only need one set of chaotic values for block
  // permutations.
  const size_t num_blocks_permutations = 1;

  int bytes_for_rows = img_dimensions.rows * 2;
  int bytes_for_columns = img_dimensions.cols * 2;
  int bytes_for_blocks = num_blocks_permutations * 4;
  int bytes_for_flow = img_dimensions.cols * 4;

  // Total length
  int length_bytes =
      bytes_for_rows + bytes_for_columns + bytes_for_blocks + bytes_for_flow;

  if (verbose)
    std::cout << "Password length" << std::endl
              << "Row bytes: " << bytes_for_rows << std::endl
              << "Columns bytes: " << bytes_for_columns << std::endl
              << "Blocks bytes: " << bytes_for_blocks << std::endl
              << "Flow bytes: " << bytes_for_flow << std::endl
              << "Total bytes: " << length_bytes << std::endl;

  std::vector<unsigned char> password = generate_hash(input, length_bytes);

  std::vector<std::vector<unsigned char>> password_segments(3);

  // construct segments (all sizes in bytes)
  password_segments[0] = std::vector<unsigned char>(
      password.begin(), password.begin() + bytes_for_rows); // Rows
  password_segments[1] = std::vector<unsigned char>(
      password.begin() + bytes_for_rows,
      password.begin() + bytes_for_rows + bytes_for_columns); // Columns
  password_segments[2] = std::vector<unsigned char>(
      password.begin() + bytes_for_rows + bytes_for_columns,
      password.end()); // Blocks and flow
  return password_segments;
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

// Dummy kernel for warmup
__global__ void warmup_kernel() { return; }

void warmup_gpu() {
  cudaFree(0); // Initialize context
  warmup_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}