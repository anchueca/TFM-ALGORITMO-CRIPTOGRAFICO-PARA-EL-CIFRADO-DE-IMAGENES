/**
 * @file aux.cu
 * @brief Helper implementations for image stacking, unstacking and password
 * derivation.
 */

#include "../include/aux.cuh"

// Generate SHA3-512-derived bytes (implementation; see header for API)
__host__ std::vector<unsigned char> generate_hash(const std::string &input, size_t length) {
    
    // 1. Crear el contexto de OpenSSL
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx == nullptr) {
        throw std::runtime_error("Error: Fallo al crear el contexto EVP de OpenSSL");
    }

    // 2. Inicializar el digest para SHAKE256
    // Importante: SHAKE es un XOF (Extendable Output Function)
    if (EVP_DigestInit_ex(ctx, EVP_shake256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Error: Fallo al inicializar SHAKE256");
    }

    // 3. Absorb (Alimentar datos)
    if (EVP_DigestUpdate(ctx, input.data(), input.size()) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Error: Fallo al actualizar el digest (Update)");
    }

    // 4. Preparar el vector de salida del tamaño solicitado (length)
    std::vector<unsigned char> output(length);

    // 5. Squeeze (Extraer datos)
    // Para SHAKE se usa EVP_DigestFinalXOF, no EVP_DigestFinal_ex
    if (EVP_DigestFinalXOF(ctx, output.data(), length) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Error: Fallo al extraer el hash (FinalXOF)");
    }

    EVP_MD_CTX_free(ctx);

    return output;
}

// Calculate password segments from a master password (implementation)
__host__ std::vector<std::vector<unsigned char>>
calculate_password(const std::string &input, size_t num_blocks,
                   Image_dimensions img_dimensions, bool verbose) {

  // Required lengths
  int bytes_for_rows = img_dimensions.rows * 2;
  int bytes_for_columns = img_dimensions.cols * 2;
  int bytes_for_blocks = num_blocks * 4;
  int bytes_for_flow = img_dimensions.cols * 4;

  // Total length
  int length_bytes =
      bytes_for_rows + bytes_for_columns + bytes_for_blocks + bytes_for_flow;

  if(verbose) std::cout << "Password lenght" << std::endl
  << "Row bytes: " << bytes_for_rows << std::endl
  << "Columns bytes: " << bytes_for_columns << std::endl
  << "BLocks bytes: " << bytes_for_blocks << std::endl
  << "Flow bytes: " << bytes_for_flow << std::endl
  << "Total bytes: " << length_bytes << std::endl;

  std::vector<unsigned char> password = generate_hash(input, length_bytes);

  std::vector<std::vector<unsigned char>> password_segments(3);

  // construct segments (all sizes in bytes)
  password_segments[0] = std::vector<unsigned char>(
      password.begin(), password.begin() + bytes_for_rows); //Rows  
  password_segments[1] = std::vector<unsigned char>(
      password.begin() + bytes_for_rows,
      password.begin() + bytes_for_rows + bytes_for_columns); //Columns
  password_segments[2] = std::vector<unsigned char>(
      password.begin() + bytes_for_rows + bytes_for_columns,
      password.end()); //Blocks and flow
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