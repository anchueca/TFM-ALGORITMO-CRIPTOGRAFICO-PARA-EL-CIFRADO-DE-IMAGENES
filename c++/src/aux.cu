#include "../include/aux.cuh"

__host__ cv::Mat unstack_image(cv::Mat image){
    int width = image.cols;
    int height = image.rows;

    cv::Mat unstacked_image(height, width * 3, CV_8UC1);

    unsigned char *d_src, *d_dst;

    cudaMalloc(&d_src, width * height * 3 * sizeof(unsigned char));
    cudaMalloc(&d_dst, width * height * 3 * sizeof(unsigned char));

    cudaMemcpy(d_src, image.data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    split_and_concat_kernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, height);

    cudaMemcpy(unstacked_image.data, d_dst, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);

    return unstacked_image;
}

__host__ cv::Mat stack_image(cv::Mat image) {
    int dst_width = image.cols / 3;
    int dst_height = image.rows;

    cv::Mat stacked_image(dst_height, dst_width, CV_8UC3);
    
    unsigned char *d_src, *d_dst;
    
    cudaMalloc(&d_src, image.total() * image.elemSize());
    cudaMalloc(&d_dst, stacked_image.total() * stacked_image.elemSize());

    cudaMemcpy(d_src, image.data, image.total() * image.elemSize(), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((dst_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (dst_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    merge_and_stack_kernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, dst_width, dst_height);

    cudaMemcpy(stacked_image.data, d_dst, stacked_image.total() * stacked_image.elemSize(), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);

    return stacked_image;
}

__host__ std::vector<unsigned char> generate_sha3_hash(const std::string &input, size_t length) {
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

/// @brief Generate a password bitstream 
/// @param input password
/// @param num_blocks 
/// @param precision_level 
/// @param rounds 
/// @param image_height 
/// @param image_width 
/// @return a vector of bitsream. Rows, cols, blocks and flow
__host__ std::vector<std::vector<unsigned char>> calculate_password(const std::string &input, int num_blocks, int precision_level, int rounds, int image_height, int image_width){

    // Required lengths
    int bytes_for_rows = image_height * precision_level;
    int bytes_for_columns = image_width * precision_level;
    int bytes_for_blocks = num_blocks * precision_level;
    int bytes_for_flow = image_width * precision_level;

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
    password_segments[3] = std::vector<unsigned char>(password.begin() + bytes_for_rows + bytes_for_columns + bytes_for_blocks, password.end());
    return password_segments;
}
