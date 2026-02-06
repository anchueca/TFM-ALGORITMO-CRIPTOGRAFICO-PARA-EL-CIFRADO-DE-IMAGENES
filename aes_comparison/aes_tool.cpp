#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <openssl/evp.h>
#include <vector>

void handleErrors() {
  std::cerr << "An error occurred using OpenSSL." << std::endl;
  exit(1);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <enc|dec> <ecb|cbc|ctr> <input_file> <output_file>"
              << std::endl;
    return 1;
  }

  std::string action = argv[1];
  std::string mode = argv[2];
  const char *inputFile = argv[3];
  const char *outputFile = argv[4];

  bool encrypting = (action == "enc");

  // Select cipher mode
  const EVP_CIPHER *cipher;
  if (mode == "ecb") {
    cipher = EVP_aes_256_ecb();
  } else if (mode == "cbc") {
    cipher = EVP_aes_256_cbc();
  } else if (mode == "ctr") {
    cipher = EVP_aes_256_ctr();
  } else {
    std::cerr << "Invalid mode. Use 'ecb', 'cbc', or 'ctr'." << std::endl;
    return 1;
  }

  // Read input file
  std::ifstream instream(inputFile, std::ios::binary);
  if (!instream) {
    std::cerr << "Could not open input file: " << inputFile << std::endl;
    return 1;
  }
  std::vector<unsigned char> inputData(
      (std::istreambuf_iterator<char>(instream)),
      std::istreambuf_iterator<char>());
  instream.close();

  // Fixed Key and IV for benchmarking/comparison purposes (32 bytes key, 16
  // bytes IV)
  unsigned char key[32] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                           0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
                           0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
                           0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20};
  unsigned char iv[16] = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
                          0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30};

  // Prepare output buffer
  std::vector<unsigned char> outputData(inputData.size() +
                                        EVP_MAX_BLOCK_LENGTH);
  int len;
  int output_len;

  EVP_CIPHER_CTX *ctx;
  if (!(ctx = EVP_CIPHER_CTX_new()))
    handleErrors();

  auto start = std::chrono::high_resolution_clock::now();

  if (encrypting) {
    if (1 != EVP_EncryptInit_ex(ctx, cipher, NULL, key, iv))
      handleErrors();
    if (1 != EVP_EncryptUpdate(ctx, outputData.data(), &len, inputData.data(),
                               inputData.size()))
      handleErrors();
    output_len = len;
    if (1 != EVP_EncryptFinal_ex(ctx, outputData.data() + len, &len))
      handleErrors();
    output_len += len;
  } else {
    if (1 != EVP_DecryptInit_ex(ctx, cipher, NULL, key, iv))
      handleErrors();
    if (1 != EVP_DecryptUpdate(ctx, outputData.data(), &len, inputData.data(),
                               inputData.size()))
      handleErrors();
    output_len = len;
    if (1 != EVP_DecryptFinal_ex(ctx, outputData.data() + len, &len))
      handleErrors();
    output_len += len;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;

  EVP_CIPHER_CTX_free(ctx);

  // Standardized output for parsing
  std::cout << "MODE: " << mode << std::endl;
  std::cout << "ACTION: " << (encrypting ? "Encryption" : "Decryption")
            << std::endl;
  std::cout << "TIME: " << duration.count() << " ms" << std::endl;
  std::cout << "SIZE: " << inputData.size() << " bytes" << std::endl;

  // Write output file
  std::ofstream outstream(outputFile, std::ios::binary);
  if (!outstream) {
    std::cerr << "Could not open output file: " << outputFile << std::endl;
    return 1;
  }
  outstream.write(reinterpret_cast<const char *>(outputData.data()),
                  output_len);
  outstream.close();

  return 0;
}
