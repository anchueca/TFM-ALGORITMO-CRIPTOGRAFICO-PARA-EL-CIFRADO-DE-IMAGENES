#ifndef VIDEO_CRYPTO_CUH
#define VIDEO_CRYPTO_CUH

#include <string>
#include <vector>
#include <opencv2/core.hpp>

// Include the original project headers
#include "../../cuda/include/structs.cuh"
#include "../../cuda/include/aux.cuh"
#include "../../cuda/include/encryption.cuh"
#include "../../cuda/include/encryption_aux.cuh"

/**
 * @brief Streaming-oriented wrapper for the GPU encryption pipeline.
 *
 * Keeps the GPU context and key material alive across frames,
 * allowing fast per-frame encryption/decryption without
 * re-initializing CUDA for every frame.
 */
class VideoEncryptor {
public:
    /**
     * @brief Initialize the encryptor for a fixed frame size.
     *
     * @param password  User-provided password string.
     * @param width     Frame width in pixels.
     * @param height    Frame height in pixels.
     * @param channels  Number of channels (1 or 3).
     * @param encrypt   true = encrypt, false = decrypt.
     */
    VideoEncryptor(const std::string& password, int width, int height,
                   int channels, bool encrypt);

    ~VideoEncryptor();

    /**
     * @brief Process (encrypt or decrypt) a single video frame.
     *
     * @param frame      Input frame.
     * @param image_hash For encryption: returns the calculated hash.
     *                   For decryption: uses the provided hash.
     * @return Processed frame.
     */
    cv::Mat processFrame(const cv::Mat& frame, uint16_t* image_hash = nullptr);

    void setImageHash(uint16_t hash) { params_.image_hash = hash; }
    uint16_t getImageHash() const { return params_.image_hash; }

    int getPaddedWidth() const { return padded_dim_; }
    int getPaddedHeight() const { return padded_dim_; }

private:
    // Configuration
    std::string password_;
    int orig_width_;
    int orig_height_;
    int orig_channels_;
    int padded_dim_;
    bool encrypt_;

    // Pre-computed key material
    std::vector<std::vector<unsigned char>> password_segments_;

    // Encryption parameters
    EncryptionParams params_;
    Image_dimensions img_dimensions_;

    // Key material copies (for re-initialization each frame)
    std::vector<unsigned char> automata_state_host_;
    std::vector<unsigned char> seeds_host_;
    std::vector<unsigned char> r_params_host_;

    // Disable copy
    VideoEncryptor(const VideoEncryptor&) = delete;
    VideoEncryptor& operator=(const VideoEncryptor&) = delete;
};

#endif // VIDEO_CRYPTO_CUH
