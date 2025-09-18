#include "../include/aux.hpp"

void unstack_image(cv::Mat& image){
    std::vector<cv::Mat> channels(3);

    cv::split(image, channels);

    cv::hconcat(channels, image);
}

void stack_image(cv::Mat& image) {
    int width = image.cols;
    int width_per_channel = width / 3;

    cv::Mat img_b = image(cv::Rect(0, 0, width_per_channel, image.rows));
    cv::Mat img_g = image(cv::Rect(width_per_channel, 0, width_per_channel, image.rows));
    cv::Mat img_r = image(cv::Rect(2 * width_per_channel, 0, width_per_channel, image.rows));

    cv::merge(std::vector<cv::Mat>{img_b, img_g, img_r}, image);
}

std::vector<unsigned char> generate_sha3_hash(const std::string &input, size_t length) {
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    const EVP_MD *sha3 = EVP_sha3_512();  // SHA3-512 (64 bytes)

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
        EVP_DigestUpdate(mdctx_iter, reinterpret_cast<const char *>(result.data()), result.size());
        
        EVP_DigestFinal_ex(mdctx_iter, out.data(), nullptr);
        EVP_MD_CTX_free(mdctx_iter);

        result.insert(result.end(), out.begin(), out.end());

        current_length += 64;
    }

    result.resize(length);

    return result;
}

std::vector<std::vector<unsigned char>> calculate_password(const std::string &input, int num_blocks, int precision_level, int rounds, int image_height, int image_width){

    // Required lengths
    int bytes_for_rows = image_height * precision_level;
    int bytes_for_columns = image_width * precision_level;
    int bytes_for_blocks = num_blocks * precision_level;
    int bytes_for_flow = image_height * precision_level;

    // Total length
    int length_bytes = 
        bytes_for_rows
        + bytes_for_columns
        + bytes_for_blocks
        + bytes_for_flow * rounds;

    std::vector<unsigned char> password = generate_sha3_hash(input, length_bytes);

    std::vector<std::vector<unsigned char>> password_segments(4);

    // construct segments (all sizes in bytes)
    password_segments[0] = std::vector<unsigned char>(password.begin(), password.begin() + bytes_for_rows);
    password_segments[1] = std::vector<unsigned char>(password.begin() + bytes_for_rows, password.begin() + bytes_for_rows + bytes_for_columns);
    password_segments[2] = std::vector<unsigned char>(password.begin() + bytes_for_rows + bytes_for_columns, password.begin() + bytes_for_rows + bytes_for_columns + bytes_for_blocks);
    password_segments[3] = std::vector<unsigned char>(password.begin() + bytes_for_rows + bytes_for_columns + bytes_for_flow, password.end());
    return password_segments;
}
