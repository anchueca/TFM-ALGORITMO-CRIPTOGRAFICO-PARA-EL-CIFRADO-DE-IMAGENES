/**
 * @file video_crypto.cu
 * @brief Streaming wrapper around the existing GPU encryption pipeline.
 */

#include "video_crypto.cuh"

// ─── Constructor ────────────────────────────────────────────────────────────

VideoEncryptor::VideoEncryptor(const std::string &password, int width,
                               int height, int channels, bool encrypt)
    : password_(password), orig_width_(width), orig_height_(height),
      orig_channels_(channels), encrypt_(encrypt) {

  params_.rounds = 1;
  params_.block_size = 8;
  params_.automata_steps = 20;
  params_.transition_length = 10;

  // Initial dummy hash
  params_.image_hash = 0;

  cudaFree(0);
  warmup_gpu();

  int unstacked_width = (channels == 3) ? width * 3 : width;
  int bytesNeeded = 5;
  long totalPixels = (long)unstacked_width * height;
  int minS = (int)std::ceil(std::sqrt(totalPixels + bytesNeeded));
  padded_dim_ = ((minS + params_.block_size - 1) / params_.block_size) *
                params_.block_size;

  img_dimensions_.cols = static_cast<size_t>(padded_dim_);
  img_dimensions_.rows = static_cast<size_t>(padded_dim_);

  password_segments_ =
      calculate_password(password, img_dimensions_, false, false);

  std::cerr << "[VideoEncryptor] Initialized: " << width << "x" << height
            << " ch=" << channels << " padded=" << padded_dim_ << "x"
            << padded_dim_ << " mode=" << (encrypt ? "ENCRYPT" : "DECRYPT")
            << std::endl;
}

VideoEncryptor::~VideoEncryptor() {}

// ─── Process a single frame ─────────────────────────────────────────────────

cv::Mat VideoEncryptor::processFrame(const cv::Mat &frame,
                                     uint16_t *recovery_info) {
  cv::Mat processed;

  if (encrypt_) {
    // ── ENCRYPTION FLOW ──
    processed = unstack_channels(frame, false);
    processed = padImageToSquare(processed, params_.block_size, orig_channels_);

    // 1. Calculate plaintext hash
    params_.image_hash = calculate_image_hash(processed, 2);

    // 2. Encrypt
    std::vector<std::vector<unsigned char>> frame_password = password_segments_;
    Image_dimensions dims = {(size_t)processed.cols, (size_t)processed.rows};
    encrypt_image(processed, frame_password, dims, params_, false, true);

    // 3. Embed hash into encrypted pixels
    std::vector<bool> stego_key;
    for (unsigned char b : password_segments_[3]) {
      for (int i = 0; i < 8; ++i)
        stego_key.push_back((b >> i) & 1);
    }
    std::vector<bool> msg;
    for (int i = 0; i < 16; ++i)
      msg.push_back((params_.image_hash >> i) & 1);

    std::vector<bool> R = embed_message_caos(processed, msg, stego_key);

    if (recovery_info) {
      uint16_t r_val = 0;
      for (size_t i = 0; i < 16 && i < R.size(); ++i)
        if (R[i])
          r_val |= (1 << i);
      *recovery_info = r_val;
    }
    return processed;

  } else {
    // ── DECRYPTION FLOW ──
    processed = frame.clone(); // frame is the received encrypted image

    // 1. Extract hash and restore LSBs
    if (recovery_info) {
      std::vector<bool> stego_key;
      for (unsigned char b : password_segments_[3]) {
        for (int i = 0; i < 8; ++i)
          stego_key.push_back((b >> i) & 1);
      }
      std::vector<bool> r_bits;
      for (int i = 0; i < 16; ++i)
        r_bits.push_back((*recovery_info >> i) & 1);

      std::vector<bool> msg =
          extract_message_caos(processed, r_bits, stego_key);

      uint16_t h_val = 0;
      for (size_t i = 0; i < 16 && i < msg.size(); ++i)
        if (msg[i])
          h_val |= (1 << i);
      params_.image_hash = h_val;
    }

    // 2. Decrypt using extracted hash
    std::vector<std::vector<unsigned char>> frame_password = password_segments_;
    Image_dimensions dims = {(size_t)processed.cols, (size_t)processed.rows};
    encrypt_image(processed, frame_password, dims, params_, false, false);

    // 3. Post-process (unpad and stack)
    int retrieved_channels = 1;
    processed = unpadFromSquare(processed, &retrieved_channels);

    cv::Mat output;
    if (retrieved_channels == 3) {
      int w = processed.cols / 3;
      int h = processed.rows;
      cv::Mat b = processed(cv::Rect(0, 0, w, h));
      cv::Mat g = processed(cv::Rect(w, 0, w, h));
      cv::Mat r = processed(cv::Rect(2 * w, 0, w, h));
      std::vector<cv::Mat> channels = {b, g, r};
      cv::merge(channels, output);
    } else {
      output = processed;
    }
    return output;
  }
}
