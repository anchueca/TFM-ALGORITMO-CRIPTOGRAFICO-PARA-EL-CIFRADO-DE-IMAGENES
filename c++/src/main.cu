/**
 * @file main.cu
 * @brief Command-line entry point for image encryption/decryption using the
 * CUDA pipeline.
 */

#include "../include/encryption.cuh"

#include <chrono>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

// Main CLI program (implementation).
int main(int argc, char **argv) {
  const size_t required_args = 11;
  if (argc != required_args) {
    cerr << "Invalid number of arguments! " << argc << " received. "
         << required_args << " required." << endl;
    return -1;
  }
  EncryptionParams params;

  string input_image_path = argv[1];
  string password = argv[3];
  string output_image_path = argv[2];
  params.rounds = stoi(argv[4]);

  bool verbose = strcmp(argv[5], "1") == 0;
  bool encrypt = strcmp(argv[6], "1") == 0;

  params.block_size = stoi(argv[7]);
  params.precision_level = stoi(argv[8]);
  params.automata_steps = stoi(argv[9]);
  params.transition_length = stoi(argv[10]);

  cv::Mat image = cv::imread(input_image_path);
  if (image.empty()) {
    cerr << "Could not open or find the image!" << endl;
    return -1;
  }

  int channels = image.channels();
  if (verbose) {
    std::cout << "=== Image parameters ===" << std::endl;
    std::cout << "Depth: " << image.depth() << std::endl;
    std::cout << "ElemSize: " << image.elemSize() << std::endl;
    std::cout << "input_image_path: " << input_image_path << std::endl;
    std::cout << "output_image_path: " << output_image_path << std::endl;
    std::cout << "Channels: " << channels << std::endl;
    std::cout << image.rows << "x" << image.cols << std::endl;
    std::cout << "===========================" << std::endl << std::endl;
  }

  if (channels != 1)
    image = unstack_image(image);

  auto start = std::chrono::high_resolution_clock::now();
  encrypt_image(image, password, params, verbose, encrypt);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;

  if (verbose)
    std::cout << "Encryption time: " << time.count() << " s" << std::endl;

  if (channels != 1)
    image = stack_image(image);

  if (image.empty()) {
    cerr << "Encryption failed!" << endl;
    return -1;
  }

  // cv::imshow("Encrypted Image", image);
  // cv::waitKey(0);
  cv::imwrite(output_image_path, image);

  return 0;
}