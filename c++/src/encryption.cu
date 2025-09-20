#include "../include/encryption.cuh"

__host__ void encrypt_image(cv::Mat image, const std::string& password, int rounds, int verbose) {
    int block_size = 16;
    int precision_level = 2;
    // For now we assume the image dimensions are multiples of block_size
    int num_blocks_per_row = image.rows / block_size + (image.rows % block_size != 0);
    int num_blocks_per_col = image.cols / block_size + (image.cols % block_size != 0);
    int num_blocks = num_blocks_per_row * num_blocks_per_col;
    int block_data_length = block_size*block_size;

    std::vector<std::vector<unsigned char>> password_segments = calculate_password(password, num_blocks, precision_level, rounds, image.rows, image.cols);

    std::cout<< "Block size: " << block_size << std::endl;
    std::cout<< "Num blocks per row: " << num_blocks_per_row << std::endl;
    std::cout<< "Num blocks per col: " << num_blocks_per_col << std::endl;
    std::cout<< "Num blocks: " << num_blocks << std::endl;
    std::cout<< "Block data length: " << block_data_length << std::endl;
    std::cout<< "Password segment size: " << password_segments[3].size() << std::endl;
    std::cout<< image.rows << "x" << image.cols << std::endl;

    //Generate permutations
    std::vector<std::vector<int>> permutations =
        generate_permutations(password_segments[3],block_data_length, num_blocks);
    
    std::vector<int> permutation_cols =
        generate_permutations(password_segments[0],image.cols, 1)[0];

    std::vector<int> permutation_rows =
        generate_permutations(password_segments[1],image.rows, 1)[0];

    std::cout << std::endl << "Permutations rows: " << std::endl;
    for (int i = 0; i < block_data_length; i++) {
        std::cout << permutation_rows[i] << " ";
    }
    std::cout << std::endl << "Permutations cols: " << std::endl;
    for (int i = 0; i < block_data_length; i++) {
        std::cout << permutation_cols[i] << " ";
    }

    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < block_data_length; i++) {
            std::cout << permutations[b][i] << " ";
        }
        std::cout << std::endl << "Permutations: " << std::endl;
    }

    unsigned char* d_image = nullptr;
    unsigned char* d_image_out = nullptr;
    size_t img_size = image.total() * image.elemSize();

    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_image_out, img_size);

    cudaMemcpy(d_image, image.data, img_size, cudaMemcpyHostToDevice);

    for (int i=0;i<rounds;i++){
        //Rows an columns
        rows_and_columns_permutation(d_image,d_image_out, permutation_rows, permutation_cols, image.cols, image.rows, true);

        unsigned char* temp = d_image;
        d_image = d_image_out;
        d_image_out = temp;
        
        //Block

        block_phase_permutation(d_image,d_image_out, permutations, image.cols, image.rows, block_size);

        temp = d_image;
        d_image = d_image_out;
        d_image_out = temp;
    }

    
    cudaMemcpy(image.data, d_image, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_image_out);
}

/// @brief Generate a set of permutations based on chaotic sequences derived from block passwords.
/// @param block_passwords Vector de contraseñas de los bloques.
/// @param block_length La longitud de los bloques.
/// @param num_blocks El número de bloques.
/// @return Un vector de vectores de enteros que representa las permutaciones generadas.
__host__ std::vector<std::vector<int>> generate_permutations(const std::vector<unsigned char> block_passwords, size_t block_length, size_t num_blocks)
{
    if (block_passwords.size() < num_blocks) {
        throw std::runtime_error("Insufficient passwords for blocks");
    }

    std::vector<std::vector<int>> indices(num_blocks, std::vector<int>(block_length));
    
    size_t total_size = num_blocks * block_length;
    
    unsigned char *d_passwords = nullptr;
    int *d_indices = nullptr;
    double *d_chaotic_values = nullptr;
    
    cudaError_t err = cudaMalloc(&d_passwords, num_blocks * sizeof(unsigned char));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for passwords");
    }
    
    err = cudaMalloc(&d_indices, total_size * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        throw std::runtime_error("Failed to allocate device memory for indices");
    }
    
    err = cudaMalloc(&d_chaotic_values, total_size * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_indices);
        throw std::runtime_error("Failed to allocate device memory for chaotic values");
    }
    
    // Copy
    err = cudaMemcpy(d_passwords, block_passwords.data(), num_blocks * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_indices);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("Failed to copy passwords to device");
    }
    
    const int threadsPerBlock = 256;
    const int numBlocks = (num_blocks + threadsPerBlock - 1) / threadsPerBlock;
    double r = 0.998;
    int transition_length = 20;
    
    generate_chaotic<<<numBlocks, threadsPerBlock>>>(
        d_passwords, num_blocks, d_chaotic_values, d_indices, r, block_length, transition_length
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_indices);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("Kernel execution failed");
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_indices);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("Kernel synchronization failed");
    }
    
    // return values
    std::vector<int> flat_indices(total_size);
    err = cudaMemcpy(flat_indices.data(), d_indices, total_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_passwords);
        cudaFree(d_indices);
        cudaFree(d_chaotic_values);
        throw std::runtime_error("Failed to copy results from device");
    }
    
    cudaFree(d_passwords);
    cudaFree(d_indices);
    cudaFree(d_chaotic_values);
    
    for (int i = 0; i < num_blocks; ++i) {
        std::copy(
            flat_indices.begin() + i * block_length,
            flat_indices.begin() + (i + 1) * block_length,
            indices[i].begin()
        );
    }
    
    return indices;
}


__host__ void block_phase_permutation(
    unsigned char* d_image, unsigned char* d_image_out, std::vector<std::vector<int>> &block_permutations, size_t cols, size_t rows, size_t block_size)
{
    int *d_permutations = nullptr;
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t permutations_size = block_permutations[0].size() * block_permutations.size() * sizeof(int);
    std::vector<int> flat_block_permutations;
    flat_block_permutations.reserve(block_permutations.size() * block_permutations[0].size());
    for (const auto& perm : block_permutations) {
        flat_block_permutations.insert(flat_block_permutations.end(), perm.begin(), perm.end());
    }

    cudaMalloc(&d_permutations, permutations_size);

    cudaMemcpy(d_permutations, flat_block_permutations.data(), permutations_size, cudaMemcpyHostToDevice);

    permute_blocks_kernel<<<numBlocks, threadsPerBlock>>>(
        d_image, d_image_out, d_permutations, block_size, cols, rows);

    cudaDeviceSynchronize();

    cudaFree(d_permutations);

}


__host__ void rows_and_columns_permutation(
    unsigned char* d_image, unsigned char* d_image_out, std::vector<int> &row_permutation, std::vector<int> &col_permutation, size_t cols, size_t rows,bool inverse)
{
    int *d_row_permutations = nullptr;
    int *d_col_permutations = nullptr;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t row_permutations_size = row_permutation.size() * sizeof(int);
    size_t col_permutations_size = col_permutation.size() * sizeof(int);

    cudaMalloc(&d_row_permutations, row_permutations_size);
    cudaMalloc(&d_col_permutations, col_permutations_size);

    cudaMemcpy(d_row_permutations, row_permutation.data(), row_permutations_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_permutations, col_permutation.data(), col_permutations_size, cudaMemcpyHostToDevice);

    if (inverse) {
        permute_rows_kernel<<<numBlocks, threadsPerBlock>>>(
            d_image, d_image_out, d_row_permutations, cols, rows);

        cudaDeviceSynchronize();

        permute_columns_kernel<<<numBlocks, threadsPerBlock>>>(
            d_image, d_image_out, d_col_permutations, cols, rows);

        } else{
        permute_columns_kernel<<<numBlocks, threadsPerBlock>>>(
            d_image, d_image_out, d_col_permutations, cols, rows);
            
            cudaDeviceSynchronize();
            
        permute_rows_kernel<<<numBlocks, threadsPerBlock>>>(
            d_image, d_image_out, d_row_permutations, cols, rows);
            
    }

    cudaDeviceSynchronize();

    cudaFree(d_row_permutations);
    cudaFree(d_col_permutations);
}

int main(int argc, char** argv) {
    if (argc != 6){
        cerr << "Error"<<endl;
        return -1;
    }

    string input_image_path = argv[1];
    string password = argv[2];
    int rounds = stoi(argv[3]);
    string output_image_path = argv[4];
    int verbose = stoi(argv[5]);

    cv::Mat image = cv::imread(input_image_path);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    int channels = image.channels();

    if(channels !=1) image= unstack_image(image);

    encrypt_image(image, password, rounds, verbose);

    //if(channels !=1) image= stack_image(image);

    if (image.empty()) {
        cerr << "Encryption failed!" << endl;
        return -1;
    }

    cv::imshow("Encrypted Image", image);
    cv::waitKey(0);

    return 0;
}