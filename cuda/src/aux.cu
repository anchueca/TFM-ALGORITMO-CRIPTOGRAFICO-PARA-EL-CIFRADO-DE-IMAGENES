/**
 * @file aux.cu
 * @brief Helper implementations for image stacking, unstacking and password
 * derivation.
 */

#include "../include/aux.cuh"

// Generate SHA-3 (SHAKE256) derived bytes (implementation; see header for API)
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

  int effective_threads = MAX_THREADS - 1;
  int numBlocks =
      (img_dimensions.cols + effective_threads - 1) / effective_threads;

  int bytes_for_columns = img_dimensions.cols * 2;
  int bytes_for_blocks = num_blocks_permutations * 4;

  int bytes_for_flow = (img_dimensions.cols + numBlocks) * 4;
  int bytes_for_r_params = (img_dimensions.cols + numBlocks) * 4;
  int bytes_for_stego = 8; // 64 bits for stego seed

  int length_bytes = bytes_for_columns + bytes_for_blocks + bytes_for_flow +
                     bytes_for_r_params + bytes_for_stego;

  if (verbose)
    std::cout << "[DEBUG] Key Requirements:" << std::endl
              << "  Columns bytes: " << bytes_for_columns << std::endl
              << "  Blocks bytes:  " << bytes_for_blocks << std::endl
              << "  Flow bytes:    " << bytes_for_flow << std::endl
              << "  R params bytes:" << bytes_for_r_params << std::endl
              << "  Stego bytes:   " << bytes_for_stego << std::endl
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

  std::vector<std::vector<unsigned char>> password_segments(4);

  // construct segments (all sizes in bytes)
  password_segments[0].assign(password.begin(),
                              password.begin() + bytes_for_columns); // Columns
  password_segments[1].assign(password.begin() + bytes_for_columns,
                              password.begin() + bytes_for_columns +
                                  bytes_for_blocks +
                                  bytes_for_flow); // Blocks and flow
  password_segments[2].assign(
      password.begin() + bytes_for_columns + bytes_for_blocks + bytes_for_flow,
      password.begin() + bytes_for_columns + bytes_for_blocks + bytes_for_flow +
          bytes_for_r_params); // R params
  password_segments[3].assign(password.begin() + bytes_for_columns +
                                  bytes_for_blocks + bytes_for_flow +
                                  bytes_for_r_params,
                              password.end()); // Steganography
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

#include "../include/steganography.hpp"

static std::vector<bool>
bytes_to_bits(const std::vector<unsigned char> &bytes) {
  std::vector<bool> bits;
  bits.reserve(bytes.size() * 8);
  for (unsigned char b : bytes) {
    for (int i = 0; i < 8; ++i) {
      bits.push_back((b >> i) & 1);
    }
  }
  return bits;
}

static std::vector<bool> hex_to_bits_local(const std::string &hex_str) {
  std::vector<bool> bits;
  for (size_t i = 0; i < hex_str.size(); i += 2) {
    if (i + 1 >= hex_str.size())
      break;
    std::string byte_str = hex_str.substr(i, 2);
    uint8_t byte = (uint8_t)std::stoi(byte_str, nullptr, 16);
    for (int j = 0; j < 8; ++j) {
      bits.push_back((byte >> j) & 1);
    }
  }
  return bits;
}

unsigned short extract_message_caos(cv::Mat &image,
                                    const std::vector<unsigned char> &stego_key,
                                    const std::string &input_path,
                                    const std::string &exif_hex) {
  std::vector<bool> key_bits = bytes_to_bits(stego_key);
  std::vector<bool> msg_bits;

  if (!exif_hex.empty()) {
    std::vector<bool> recovery_bits = hex_to_bits_local(exif_hex);
    msg_bits = extract_message_caos(image, recovery_bits, key_bits);
  } else {
    msg_bits = extract_message_caos_with_exif(image, key_bits, input_path);
  }

  if (msg_bits.size() < 16)
    return 0;

  unsigned short hash = 0;
  for (int i = 0; i < 16; ++i) {
    if (msg_bits[i])
      hash |= (1 << i);
  }
  return hash;
}

void embed_message_caos(cv::Mat &image, unsigned short image_hash,
                        const std::vector<unsigned char> &stego_key,
                        const std::string &output_path) {
  std::vector<bool> key_bits = bytes_to_bits(stego_key);
  std::vector<bool> msg_bits(16);
  for (int i = 0; i < 16; ++i) {
    msg_bits[i] = (image_hash >> i) & 1;
  }
  embed_message_caos_with_exif(image, msg_bits, key_bits, output_path);
}
cv::Mat padImageToSquare(const cv::Mat &input, int blockSize,
                         int original_channels) {
  if (input.cols > 65535 || input.rows > 65535) {
    throw std::runtime_error("Dimensiones exceden el límite de 16 bits.");
  }

  uint16_t W = static_cast<uint16_t>(input.cols);
  uint16_t H = static_cast<uint16_t>(input.rows);
  int channels = input.channels();

  long totalPixelsOriginal = input.total();
  int bytesNeeded =
      5; // 2 bytes for W + 2 bytes for H + 1 byte for original_channels
  int minS = std::ceil(std::sqrt(totalPixelsOriginal + bytesNeeded));
  int S = ((minS + blockSize - 1) / blockSize) * blockSize;

  cv::Mat squared = cv::Mat::zeros(S, S, input.type());
  cv::Mat flatInput = input.reshape(channels, 1);
  cv::Mat flatOutput = squared.reshape(channels, 1);
  flatInput.copyTo(flatOutput.colRange(0, totalPixelsOriginal));

  uchar *dataPtr = squared.data;
  size_t lastByteIdx = (size_t)S * S * channels;

  // Guardar metadatos en big-endian para robustez multiplataforma
  dataPtr[lastByteIdx - 5] = ((W >> 8) & 0xFF); // W High (big-endian)
  dataPtr[lastByteIdx - 4] = (W & 0xFF);        // W Low
  dataPtr[lastByteIdx - 3] = ((H >> 8) & 0xFF); // H High (big-endian)
  dataPtr[lastByteIdx - 2] = (H & 0xFF);        // H Low
  // Color byte: 1 = color, 0 = grayscale
  dataPtr[lastByteIdx - 1] = (original_channels == 3) ? 1 : 0;

  return squared;
}

cv::Mat unpadFromSquare(const cv::Mat &squared, int *out_original_channels) {
  int channels = squared.channels();
  uchar *dataPtr = squared.data;
  size_t lastByteIdx = (size_t)squared.total() * channels;

  // Leer metadatos en big-endian
  uint16_t W = (static_cast<uint16_t>(dataPtr[lastByteIdx - 5]) << 8) |
               dataPtr[lastByteIdx - 4];
  uint16_t H = (static_cast<uint16_t>(dataPtr[lastByteIdx - 3]) << 8) |
               dataPtr[lastByteIdx - 2];
  // Color byte: 1 = color, 0 = grayscale
  uchar is_color_flag = dataPtr[lastByteIdx - 1];
  int original_channels = (is_color_flag == 1) ? 3 : 1;

  // Validation: Check for corrupted metadata (common sign of decryption
  // failure)
  size_t required_pixels = (size_t)W * H;
  if (required_pixels == 0 || required_pixels > squared.total() || W == 0 ||
      H == 0) {
    throw std::runtime_error(
        "Decryption failed: Recovered image dimensions (" + std::to_string(W) +
        "x" + std::to_string(H) +
        ") are invalid or exceed buffer size. Metadata might be corrupted.");
  }

  if (out_original_channels != nullptr) {
    *out_original_channels = original_channels;
  }

  cv::Mat output = cv::Mat(H, W, squared.type());

  cv::Mat flatSquared = squared.reshape(channels, 1);
  cv::Mat flatOutput = output.reshape(channels, 1);
  flatSquared.colRange(0, (size_t)W * H).copyTo(flatOutput);

  return output;
}