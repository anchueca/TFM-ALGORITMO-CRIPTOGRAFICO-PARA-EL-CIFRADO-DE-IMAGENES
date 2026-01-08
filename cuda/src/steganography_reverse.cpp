#include "../include/steganography.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <libexif/exif-data.h>
#include <libexif/exif-entry.h>
#include <libexif/exif-tag.h>
#include <libexif/exif-ifd.h>

static double bits_to_seed(const std::vector<bool> &bits) {
  if (bits.empty())
    return 0.5;
  uint64_t val = 0;
  for (size_t i = 0; i < bits.size() && i < 64; ++i) {
    if (bits[i])
      val |= (1ULL << i);
  }
  double d = (double)val / (double)0xFFFFFFFFFFFFFFFFULL;
  if (d <= 0.0)
    d = 0.123456789;
  if (d >= 1.0)
    d = 0.987654321;
  return d;
}

static double logistic_map(double x, double r = 3.999) {
  return r * x * (1.0 - x);
}

/**
 * @brief Cosine-Cosine chaotic function (same as used in encryption).
 * Matches the chaotic_functio used in the main encryption pipeline.
 */
static double chaotic_cosine(double x, double r) {
  double t = r + 3.0 * x * x;
  return fabs(cos(M_PI * r * cos(M_PI * t) * t));
}

/**
 * @brief Converts a hex string back to vector<bool>.
 */
static std::vector<bool> hex_to_bits(const std::string &hex_str) {
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

/**
 * @brief Read recovery information from EXIF metadata in the image file.
 */
static std::vector<bool> read_recovery_metadata(const std::string &input_path) {
  ExifData *exif_data = exif_data_new_from_file(input_path.c_str());
  if (!exif_data) {
    std::cerr << "Warning: Could not read EXIF data from " << input_path << "\n";
    return std::vector<bool>();
  }

  std::vector<bool> recovery_bits;
  ExifEntry *entry = 
      exif_content_get_entry(exif_data->ifd[EXIF_IFD_EXIF], EXIF_TAG_USER_COMMENT);
  
  if (entry && entry->data) {
    std::string recovery_hex((char *)entry->data);
    recovery_bits = hex_to_bits(recovery_hex);
    std::cout << "Read recovery metadata from EXIF (hex: " << recovery_hex << ")\n";
  } else {
    std::cerr << "Warning: UserComment (recovery metadata) not found in EXIF.\n";
  }

  exif_data_unref(exif_data);
  return recovery_bits;
}

std::vector<bool> extract_message_caos(cv::Mat &image,
                                       const std::vector<bool> &recovery_info,
                                       const std::vector<bool> &key) {
  size_t N = recovery_info.size();
  size_t M = image.total() * image.elemSize();

  if (M == 0)
    return std::vector<bool>();

  std::vector<bool> message(N);
  double x = bits_to_seed(key);
  std::vector<int> H(N);

  // 1 & 2. Regenerate chaotic sequence using cosine-cosine map and quantization
  double r_param = 2.5; // Same chaotic parameter as in embedding
  for (size_t i = 0; i < N; ++i) {
    x = chaotic_cosine(x, r_param);
    H[i] = (int)(x * 255.0);
  }

  // 3. Same initial position p0
  long long p0 = 1;
  bool has_nonzero = false;
  for (int h : H) {
    if (h != 0) {
      p0 = (p0 * h) % M;
      has_nonzero = true;
    }
  }
  if (!has_nonzero)
    p0 = 0;

  cv::Mat flat;
  if (image.isContinuous()) {
    flat = image;
  } else {
    flat = image.clone();
  }
  uint8_t *data = flat.ptr<uint8_t>(0);

  // 4. Extraction and restoration of original image
  long long current_p = p0;
  for (size_t n = 0; n < N; n++) {
    // Redraw path
    current_p = (current_p + H[n] + 1) % M;

    uint8_t byte_val = data[current_p];
    bool current_lsb = byte_val & 1;

    // The hidden bit is the LSB of the modified image
    message[n] = current_lsb;

    // Restore original LSB: original = current ^ recovery
    bool original_lsb = current_lsb ^ recovery_info[n];
    data[current_p] = (byte_val & 0xFE) | (original_lsb ? 1 : 0);
  }

  if (!image.isContinuous()) {
    flat.copyTo(image);
  }

  return message;
}

/**
 * @brief Wrapper function that reads recovery info from EXIF and extracts message.
 */
std::vector<bool> extract_message_caos_with_exif(cv::Mat &image,
                                                 const std::vector<bool> &key,
                                                 const std::string &input_path) {
  // Read recovery information from EXIF metadata
  std::vector<bool> recovery_info = read_recovery_metadata(input_path);
  
  if (recovery_info.empty()) {
    std::cerr << "Error: Could not read recovery metadata from EXIF.\n";
    return std::vector<bool>();
  }
  
  // Extract the message using standard function
  return extract_message_caos(image, recovery_info, key);
}
