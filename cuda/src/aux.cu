/**
 * @file aux.cu
 * @brief Helper implementations for image stacking, unstacking and password
 * derivation.
 */

#include "../include/aux.cuh"

// Generate SHA3-512-derived bytes (implementation; see header for API)
// Generate hash from string (calls buffer version)
// Generate hash from buffer (implementation)
__host__ std::vector<unsigned char>
generate_hash(const unsigned char *input, size_t input_len, size_t length) {
  // 1. Create OpenSSL context
  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  if (ctx == nullptr) {
    throw std::runtime_error("Error: Failed to create OpenSSL EVP context");
  }

  // 2. Initialize digest for SHAKE256
  if (EVP_DigestInit_ex(ctx, EVP_shake256(), nullptr) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("Error: Failed to initialize SHAKE256");
  }

  // 3. Absorb (Feed data)
  if (EVP_DigestUpdate(ctx, input, input_len) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("Error: Failed to update digest");
  }

  // 4. Prepare output vector of requested size (length)
  std::vector<unsigned char> output(length);

  // 5. Squeeze (Extract data)
  if (EVP_DigestFinalXOF(ctx, output.data(), length) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("Error: Failed to extract hash (FinalXOF)");
  }

  EVP_MD_CTX_free(ctx);

  return output;
}

// Generate hash from string (calls buffer version)
__host__ std::vector<unsigned char> generate_hash(const std::string &input,
                                                  size_t length) {
  return generate_hash(reinterpret_cast<const unsigned char *>(input.data()),
                       input.size(), length);
}

// Helper to check if a string is binary
bool is_binary_string(const std::string &s) {
  if (s.empty())
    return false;
  for (char c : s) {
    if (c != '0' && c != '1')
      return false;
  }
  return true;
}

// Helper to convert bitstring to bytes
std::vector<unsigned char> bitstring_to_bytes(const std::string &bits) {
  std::vector<unsigned char> bytes;
  for (size_t i = 0; i < bits.length(); i += 8) {
    unsigned char byte = 0;
    for (size_t j = 0; j < 8 && (i + j) < bits.length(); ++j) {
      if (bits[i + j] == '1') {
        byte |= (1 << (7 - j));
      }
    }
    bytes.push_back(byte);
  }
  return bytes;
}

// Calculate password segments from a master password (implementation)
__host__ std::vector<std::vector<unsigned char>>
calculate_password(const std::string &input, Image_dimensions img_dimensions,
                   bool verbose, bool use_raw_key) {
  // Required lengths
  // Fixed to 1 as we only need one set of chaotic values for block
  // permutations.
  const size_t num_blocks_permutations = 1;

  int bytes_for_rows = img_dimensions.rows * 2;
  int bytes_for_columns = img_dimensions.cols * 2;
  int bytes_for_blocks = num_blocks_permutations * 4;

  // IMPORTANT: bytes_for_flow must match the allocation in
  // generate_flow_stream_parallel numBlocks is (cols + 256) / 256
  int numBlocks = (img_dimensions.cols + 256) / 256;
  int bytes_for_flow = (img_dimensions.cols + numBlocks) * 4;

  // Total length
  int length_bytes =
      bytes_for_rows + bytes_for_columns + bytes_for_blocks + bytes_for_flow;

  if (verbose)
    std::cout << "[DEBUG] Key Requirements:" << std::endl
              << "  Row bytes:     " << bytes_for_rows << std::endl
              << "  Columns bytes: " << bytes_for_columns << std::endl
              << "  Blocks bytes:  " << bytes_for_blocks << std::endl
              << "  Flow bytes:    " << bytes_for_flow << std::endl
              << "  Total bytes:   " << length_bytes << " (" << length_bytes * 8
              << " bits)" << std::endl;

  std::vector<unsigned char> password;

  // Use raw bitstring if explicitly requested
  if (use_raw_key) {
    if (input.length() != (size_t)length_bytes * 8) {
      throw std::runtime_error(
          "Error: Raw bitstring length does not match requirements. Expected " +
          std::to_string(length_bytes * 8) + " bits.");
    }
    if (verbose)
      std::cout
          << "[INFO] Raw bitstring key used as requested. Skipping SHAKE256."
          << std::endl;
    password = bitstring_to_bytes(input);
  } else {
    password = generate_hash(input, length_bytes);
  }

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

unsigned short calculate_image_hash(const cv::Mat &image, size_t length) {
  cv::Mat temp = image.isContinuous() ? image : image.clone();
  std::vector<unsigned char> h =
      generate_hash(temp.data, temp.total() * temp.elemSize(), length);
  return h.size() >= 2 ? (h[0] << 8 | h[1]) : (h.size() ? h[0] : 0);
}