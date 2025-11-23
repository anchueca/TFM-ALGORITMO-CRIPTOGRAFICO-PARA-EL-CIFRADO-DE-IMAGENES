/**
 * @file aux.cu
 * @brief Helper implementations for image stacking, unstacking and password
 * derivation.
 */

#include "../include/aux.cuh"

// Generate SHA3-512-derived bytes (implementation; see header for API)
__host__ std::vector<unsigned char> generate_sha3_hash(const std::string &input,
                                                       size_t length) {
  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  const EVP_MD *sha3 = EVP_sha3_512(); // SHA3-512 (64 bytes)

  EVP_DigestInit_ex(mdctx, sha3, nullptr);
  EVP_DigestUpdate(mdctx, input.c_str(), input.length());

  std::vector<unsigned char> out(64);
  EVP_DigestFinal_ex(mdctx, out.data(), nullptr);
  EVP_MD_CTX_free(mdctx);

  if (length <= 64) {
    out.resize(length);
    return out;
  }

  std::vector<unsigned char> result;
  size_t current_length = 64;

  result.insert(result.end(), out.begin(), out.end());

  while (current_length < length) {
    EVP_MD_CTX *mdctx_iter = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx_iter, sha3, nullptr);
    EVP_DigestUpdate(mdctx_iter, reinterpret_cast<const char *>(result.data()),
                     result.size());

    EVP_DigestFinal_ex(mdctx_iter, out.data(), nullptr);
    EVP_MD_CTX_free(mdctx_iter);

    result.insert(result.end(), out.begin(), out.end());

    current_length += 64;
  }

  result.resize(length);

  return result;
}

// Calculate password segments from a master password (implementation)
__host__ std::vector<std::vector<unsigned char>>
calculate_password(const std::string &input, size_t num_blocks,
                   size_t precision_level, Image_dimnesions img_dimensions) {

  // Required lengths
  int bytes_for_rows = img_dimensions.rows * precision_level;
  int bytes_for_columns = img_dimensions.cols * precision_level;
  int bytes_for_blocks = num_blocks * precision_level * 4;
  int bytes_for_flow = img_dimensions.cols * precision_level*4;

  // Total length
  int length_bytes =
      bytes_for_rows + bytes_for_columns + bytes_for_blocks + bytes_for_flow;

  std::vector<unsigned char> password = generate_sha3_hash(input, length_bytes);

  std::vector<std::vector<unsigned char>> password_segments(4);

  // construct segments (all sizes in bytes)
  password_segments[0] = std::vector<unsigned char>(
      password.begin(), password.begin() + bytes_for_rows);
  password_segments[1] = std::vector<unsigned char>(
      password.begin() + bytes_for_rows,
      password.begin() + bytes_for_rows + bytes_for_columns);
  password_segments[2] = std::vector<unsigned char>(
      password.begin() + bytes_for_rows + bytes_for_columns,
      password.begin() + bytes_for_rows + bytes_for_columns + bytes_for_blocks);
  password_segments[3] = std::vector<unsigned char>(
      password.begin() + bytes_for_rows + bytes_for_columns + bytes_for_blocks,
      password.end());
  return password_segments;
}
