#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

#include "../include/aux.hpp"
#include "../include/kernels.cuh"

using namespace std;

void encrypt_image(cv::Mat& image, const string& password, int rounds, int verbose) {
    int num_blocks = 256;
    int precision_level = 2;

    std::vector<std::vector<unsigned char>> password_segments = calculate_password(password, num_blocks, precision_level, rounds, image.rows, image.cols);

    // Pruebas
    std::cout << "Passwords: "<< password_segments[3].size() << std::endl;
    for(int i=0; i<password_segments[3].size();i++){
        std::cout << static_cast<int>(password_segments[3][i]) << " ";
    }
    
    int num_rows = floor(sqrt(num_blocks));
    int num_cols = floor(sqrt(num_blocks / num_rows));

    // Block size as square as possible
    int block_height = image.rows / num_rows; // num_rows
    int block_width = image.cols / num_cols; // num_cols

    int block_data_length = block_height * block_width;

    std::vector<std::vector<int>> permutations = generate_permutations(password_segments[3],block_data_length, num_blocks);
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < block_data_length; i++) {
            std::cout << permutations[b][i] << " ";
        }
        std::cout << std::endl << "Permutations: ";
    }

    //block_phase_permutation(image, num_rows, num_cols, block_height, block_width, permutations);

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

    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    unstack_image(image);

    encrypt_image(image, password, rounds, verbose);

    stack_image(image);

    if (image.empty()) {
        cerr << "Encryption failed!" << endl;
        return -1;
    }

    cv::imshow("Encrypted Image", image);
    cv::waitKey(0);

    return 0;
}