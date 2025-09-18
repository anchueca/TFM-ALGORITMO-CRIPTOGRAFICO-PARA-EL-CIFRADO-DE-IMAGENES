#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <math.h>

#include "../include/aux.hpp"
#include "../include/kernels.cuh"

using namespace std;

void encrypt_image(cv::cuda::GpuMat image, const string& password, int rounds, int verbose) {
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

    std::vector<std::vector<int>> permutations =
    generate_permutations(password_segments[3],block_data_length, num_blocks);
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < block_data_length; i++) {
            std::cout << permutations[b][i] << " ";
        }
        std::cout << std::endl << "Permutations: " << std::endl;
    }

    block_phase_permutation(image, permutations);

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

    if(channels !=1) unstack_image(image);

    cv::cuda::GpuMat d_image;
    d_image.upload(image);

    encrypt_image(d_image, password, rounds, verbose);

    d_image.download(image);

    if(channels !=1) stack_image(image);

    if (image.empty()) {
        cerr << "Encryption failed!" << endl;
        return -1;
    }

    cv::imshow("Encrypted Image", image);
    cv::waitKey(0);

    return 0;
}