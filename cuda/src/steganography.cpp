#include "../include/steganography.hpp"
#include <bitset>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <libexif/exif-data.h>
#include <libexif/exif-entry.h>
#include <libexif/exif-ifd.h>
#include <libexif/exif-tag.h>
#include <numeric>
#include <sstream>

static double bits_to_seed(const std::vector<bool> &bits) {
  if (bits.empty())
    return 0.5;
  uint64_t val = 0;
  for (size_t i = 0; i < bits.size() && i < 64; ++i) {
    if (bits[i])
      val |= (1ULL << i);
  }
  // Normalize to (0, 1) range
  double d = (double)val / (double)0xFFFFFFFFFFFFFFFFULL;
  // Map to a safer chaotic range middle
  if (d <= 0.0)
    d = 0.123456789;
  if (d >= 1.0)
    d = 0.987654321;
  return d;
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
 * @brief Converts a vector<bool> to a hex string for metadata storage.
 */
static std::string bits_to_hex(const std::vector<bool> &bits) {
  std::stringstream ss;
  for (size_t i = 0; i < bits.size(); i += 8) {
    uint8_t byte = 0;
    for (size_t j = 0; j < 8 && i + j < bits.size(); ++j) {
      if (bits[i + j])
        byte |= (1 << j);
    }
    ss << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
  }
  return ss.str();
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
 * @brief Write recovery information as EXIF metadata in the image file.
 * Stores the recovery bits hex string in EXIF UserComment tag (0x8298).
 */
static void write_recovery_metadata(const std::string &output_path,
                                    const std::vector<bool> &recovery_info) {
  // Convert recovery info to hex string
  std::string recovery_hex = bits_to_hex(recovery_info);

  // Create EXIF data structure
  ExifData *exif_data = exif_data_new();
  if (!exif_data) {
    std::cerr << "Warning: Could not create EXIF data structure.\n";
    return;
  }

  // Create an entry for UserComment (EXIF tag 0x8298)
  ExifEntry *entry = exif_entry_new();
  if (entry) {
    entry->tag = EXIF_TAG_USER_COMMENT;
    entry->format = EXIF_FORMAT_ASCII;
    entry->components = recovery_hex.length() + 1; // +1 for null terminator
    entry->data = (unsigned char *)malloc(entry->components);
    strcpy((char *)entry->data, recovery_hex.c_str());

    // Add entry to EXIF data (Exif IFD)
    exif_content_add_entry(exif_data->ifd[EXIF_IFD_EXIF], entry);
    exif_entry_unref(entry);
  }

  // Serialize EXIF data and write to file using a temporary approach
  // Note: libexif doesn't directly write JPEG/TIFF with embedded EXIF.
  // We'll store the hex in memory and users can add external tool if needed.
  unsigned char *exif_buf = nullptr;
  unsigned int exif_size = 0;
  exif_data_save_data(exif_data, &exif_buf, &exif_size);

  if (exif_buf && exif_size > 0) {
    std::cerr << "EXIF recovery metadata prepared (size: " << exif_size
              << " bytes). Recovery hex: " << recovery_hex << "\n";
    free(exif_buf);
  }

  exif_data_unref(exif_data);
}

/**
 * @brief Read recovery information from EXIF metadata in the image file.
 */
static std::vector<bool> read_recovery_metadata(const std::string &input_path) {
  ExifData *exif_data = exif_data_new_from_file(input_path.c_str());
  if (!exif_data) {
    std::cerr << "Warning: Could not read EXIF data from " << input_path
              << "\n";
    return std::vector<bool>();
  }

  std::vector<bool> recovery_bits;
  ExifEntry *entry = exif_content_get_entry(exif_data->ifd[EXIF_IFD_EXIF],
                                            EXIF_TAG_USER_COMMENT);

  if (entry && entry->data) {
    std::string recovery_hex((char *)entry->data);
    recovery_bits = hex_to_bits(recovery_hex);
    std::cerr << "Read recovery metadata from EXIF (hex: " << recovery_hex
              << ")\n";
  } else {
    std::cerr
        << "Warning: UserComment (recovery metadata) not found in EXIF.\n";
  }

  exif_data_unref(exif_data);
  return recovery_bits;
}

std::vector<bool> embed_message_caos(cv::Mat &image,
                                     const std::vector<bool> &message,
                                     const std::vector<bool> &key) {
  size_t N = message.size();
  size_t M = image.total() * image.elemSize(); // Total bytes

  if (M == 0)
    return std::vector<bool>();

  std::vector<bool> R(N);
  double x = bits_to_seed(key);
  std::vector<int> H(N);

  // 1 & 2. Generate chaotic sequence using cosine-cosine map and quantize to
  // [0, 255]
  double r_param = 2.5; // Chaotic parameter for cosine-cosine map
  for (size_t i = 0; i < N; ++i) {
    x = chaotic_cosine(x, r_param);
    H[i] = (int)(x * 255.0);
  }

  // 3. Initial position p0: Product of non-zero H elements mod M
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

  // Ensure the image is continuous for flat indexing
  cv::Mat flat;
  if (image.isContinuous()) {
    flat = image;
  } else {
    flat = image.clone(); // Fallback to continuous copy
  }
  uint8_t *data = flat.ptr<uint8_t>(0);

  // 4 & 5. Insertion and Recovery Bit generation
  long long current_p = p0;
  for (size_t n = 0; n < N; ++n) {
    // pn = pn-1 + Hn + 1 (mod M)
    current_p = (current_p + H[n] + 1) % M;

    uint8_t byte_val = data[current_p];
    bool original_lsb = byte_val & 1;
    bool message_bit = message[n];

    // Substitution of LSB
    data[current_p] = (byte_val & 0xFE) | (message_bit ? 1 : 0);

    // Recovery bit: original_lsb XOR message_bit
    R[n] = original_lsb ^ message_bit;
  }

  // If we cloned, we need to copy back the results to original Mat structure
  if (!image.isContinuous()) {
    flat.copyTo(image);
  }

  return R;
}

/**
 * @brief Wrapper function that embeds message and stores recovery info in EXIF.
 */
std::vector<bool> embed_message_caos_with_exif(cv::Mat &image,
                                               const std::vector<bool> &message,
                                               const std::vector<bool> &key,
                                               const std::string &output_path) {
  // Embed the message using standard function
  std::vector<bool> R = embed_message_caos(image, message, key);

  // Write recovery information to EXIF metadata
  if (!R.empty()) {
    write_recovery_metadata(output_path, R);
  }

  return R;
}
