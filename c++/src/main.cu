#include "../include/encryption.cuh"

#include <chrono>
#include <cstring> // Para strcmp
#include <iomanip> // Para formateo de salida
#include <iostream>
#include <stdexcept> // Para manejo de excepciones

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void print_usage(const char *prog_name) {
  cerr << "Usage: " << prog_name << " <InputPath> <OutputPath> <Password> "
       << "<Rounds> <Verbose(0/1)> <Mode(1=Enc/0=Dec)> "
       << "<BlockSize> <Precision> <AutoSteps> <TransLen>" << endl;
}

int main(int argc, char **argv) {
  const size_t required_args = 11;

  if (argc != required_args) {
    cerr << "[ERROR] Invalid number of arguments! " << argc << " received, "
         << required_args << " required." << endl;
    print_usage(argv[0]);
    return -1;
  }

  EncryptionParams params;
  string input_image_path;
  string output_image_path;
  string password;
  bool verbose = false;
  bool encrypt = true;

  // --- 1. Parse Arguments ---
  try {
    input_image_path = argv[1];
    output_image_path = argv[2];
    password = argv[3];
    params.rounds = stoi(argv[4]);

    verbose = (strcmp(argv[5], "1") == 0);
    encrypt = (strcmp(argv[6], "1") == 0);

    params.block_size = stoi(argv[7]);
    params.precision_level = stoi(argv[8]);
    params.automata_steps = stoi(argv[9]);
    params.transition_length = stoi(argv[10]);
  } catch (const std::exception &e) {
    cerr << "[ERROR] Parsing arguments failed: " << e.what() << endl;
    print_usage(argv[0]);
    return -1;
  }

  // --- 2. Load Image ---
  // IMREAD_UNCHANGED is crucial to preserve channels (Color vs Grayscale)
  // so the encrypt_image logic can detect if it needs to unstack/stack.
  cv::Mat image = cv::imread(input_image_path, cv::IMREAD_UNCHANGED);

  if (image.empty()) {
    cerr << "[ERROR] Could not open or find the image at: " << input_image_path
         << endl;
    return -1;
  }

  // --- 3. Initial Report (Main level) ---
  if (verbose) {
    std::cout
        << "\n============================================================"
        << std::endl;
    std::cout << "                     CLI EXECUTION START                    "
              << std::endl;
    std::cout << "============================================================"
              << std::endl;
    std::cout << "  Input File:   " << input_image_path << std::endl;
    std::cout << "  Output File:  " << output_image_path << std::endl;
    std::cout << "  Loaded Size:  " << image.cols << "x" << image.rows
              << std::endl;
    std::cout << "  Channels:     " << image.channels() << std::endl;
    std::cout
        << "============================================================\n"
        << std::endl;
  }

  // --- 4. Execution ---
  auto start = std::chrono::high_resolution_clock::now();

  try {
    // Call the smart encryption function
    // This handles Unstacking -> GPU Encryption -> Restacking internally.
    encrypt_image(image, password, params, verbose, encrypt);
  } catch (const std::exception &e) {
    cerr << "\n[FATAL ERROR] During encryption process: " << e.what() << endl;
    return -1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;

  if (verbose) {
    std::cout << "Total Pipeline Time: " << time.count() << " s" << std::endl;
  }

  // --- 5. Validation & Save ---
  if (image.empty()) {
    cerr << "[ERROR] Resulting image is empty. Encryption/Decryption failed."
         << endl;
    return -1;
  }

  // Save the result
  // OpenCV handles format encoding (png, jpg, etc) based on extension in
  // output_image_path
  if (!cv::imwrite(output_image_path, image)) {
    cerr << "[ERROR] Failed to write output image to: " << output_image_path
         << endl;
    return -1;
  }

  if (verbose)
    std::cout << "Image saved successfully." << std::endl;

  return 0;
}