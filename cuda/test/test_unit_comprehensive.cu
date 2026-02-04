/**
 * @file test_unit_comprehensive.cu
 * @brief Comprehensive unit test suite for encryption functions and kernels
 *
 * This file contains an extensive battery of unit tests covering:
 * - Parameter validation
 * - Kernel functionality
 * - Cipher operations (encryption/decryption)
 * - Elementary cellular automata
 * - Permutation operations
 * - Edge cases and various input configurations
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <vector>

// Project headers
#include "../include/automata.cuh"
#include "../include/encryption.cuh"
#include "../include/encryption_aux.cuh"
#include "../include/kernels.cuh"
#include "../include/structs.cuh"

// ============================================================================
// FIXTURE: Test infrastructure
// ============================================================================

class CipherTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize CUDA device
    cudaSetDevice(0);
    cudaFree(0); // Warmup
  }

  void TearDown() override {
    // Clean up CUDA memory if needed
    cudaDeviceSynchronize();
  }

  // Helper: Create random byte vector
  std::vector<unsigned char> create_random_bytes(size_t size) {
    std::vector<unsigned char> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < size; ++i) {
      data[i] = dis(gen);
    }
    return data;
  }

  // Helper: Create test image
  cv::Mat create_test_image(int width, int height, int channels = 1) {
    cv::Mat img(height, width, CV_8UC(channels));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < img.total() * channels; ++i) {
      img.data[i] = dis(gen);
    }
    return img;
  }

  // Helper: Verify CUDA error
  void check_cuda_error(cudaError_t error) {
    ASSERT_EQ(error, cudaSuccess)
        << "CUDA Error: " << cudaGetErrorString(error);
  }

  // Helper: Create proper password structure matching image dimensions
  std::vector<std::vector<unsigned char>>
  create_password_for_image(Image_dimensions dims) {
    // Calculate required sizes based on calculate_password logic
    const size_t num_blocks_permutations = 1;
    int bytes_for_columns = dims.cols * 2;
    // Rows are equal to cols due to padding, so we just use columns bytes or they are shared.
    // The implementation in aux.cu only allocates bytes_for_columns.
    int bytes_for_blocks = num_blocks_permutations * 4;
    int numBlocks = (dims.cols + 256) / 256;
    int bytes_for_flow = (dims.cols + numBlocks) * 4;
    int bytes_for_stego = 8;
    int total_bytes =
        bytes_for_columns + bytes_for_blocks + bytes_for_flow + bytes_for_stego;

    // Generate random password
    std::vector<unsigned char> password = create_random_bytes(total_bytes);

    // Split into segments
    std::vector<std::vector<unsigned char>> segments(3);
    segments[0].assign(password.begin(),
                       password.begin() + bytes_for_columns);
    segments[1].assign(password.begin() + bytes_for_columns,
                       password.begin() + bytes_for_columns + bytes_for_blocks +
                           bytes_for_flow);
    segments[2].assign(password.begin() + bytes_for_columns + bytes_for_blocks +
                           bytes_for_flow,
                       password.end());
    return segments;
  }
};

// ============================================================================
// TEST GROUP 1: Parameter Validation Tests
// ============================================================================

class ParameterValidationTest : public CipherTest {};

TEST_F(ParameterValidationTest, ValidEncryptionParams) {
  EncryptionParams params;
  params.rounds = 3;
  params.block_size = 8;
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_GT(params.rounds, 0);
  ASSERT_GT(params.block_size, 0);
  ASSERT_GT(params.automata_steps, 0);
  ASSERT_GT(params.transition_length, 0);
  ASSERT_GE(params.chaos_parameter, 3.5f);
  ASSERT_LE(params.chaos_parameter, 4.0f);
}

TEST_F(ParameterValidationTest, BlockSizePowerOfTwo) {
  std::vector<size_t> valid_sizes = {4, 8, 16, 32, 64};
  for (size_t size : valid_sizes) {
    EncryptionParams params;
    params.block_size = size;
    ASSERT_EQ(params.block_size & (params.block_size - 1), 0)
        << "Block size should be power of 2";
  }
}

TEST_F(ParameterValidationTest, ChaosParameterRange) {
  std::vector<float> valid_params = {3.6f, 3.7f, 3.8f, 3.9f, 4.0f};
  for (float r : valid_params) {
    EncryptionParams params;
    params.chaos_parameter = r;
    ASSERT_GE(params.chaos_parameter, 3.5f);
    ASSERT_LE(params.chaos_parameter, 4.0f);
  }
}

TEST_F(ParameterValidationTest, ImageDimensionsPositive) {
  Image_dimensions dims;
  dims.rows = 256;
  dims.cols = 256;

  ASSERT_GT(dims.rows, 0);
  ASSERT_GT(dims.cols, 0);
  ASSERT_GT(dims.rows * dims.cols, 0);
}

TEST_F(ParameterValidationTest, ImageDimensionsAlignedToBlockSize) {
  std::vector<size_t> block_sizes = {4, 8, 16, 32};
  std::vector<size_t> image_sizes = {32, 64, 128, 256};

  for (size_t block_size : block_sizes) {
    for (size_t size : image_sizes) {
      Image_dimensions dims;
      dims.rows = size;
      dims.cols = size;
      ASSERT_EQ(dims.rows % block_size, 0)
          << "Image dimensions should be aligned";
      ASSERT_EQ(dims.cols % block_size, 0)
          << "Image dimensions should be aligned";
    }
  }
}

// ============================================================================
// TEST GROUP 2: Elementary Cellular Automata Tests
// ============================================================================

class CellularAutomataTest : public CipherTest {};

TEST_F(CellularAutomataTest, ConstructorRandomInitialization) {
  ASSERT_NO_THROW(ElementalCelularAutomata ca(256, 30));
  ElementalCelularAutomata ca(256, 30);
  ASSERT_NE(ca.get_cuda_state(), nullptr);
}

TEST_F(CellularAutomataTest, ConstructorFromVector) {
  std::vector<unsigned int> initial_state(8, 0x12345678);
  ASSERT_NO_THROW(
      ElementalCelularAutomata ca(initial_state, 256, 30));
  ElementalCelularAutomata ca(initial_state, 256, 30);
  ASSERT_NE(ca.get_cuda_state(), nullptr);
}

TEST_F(CellularAutomataTest, ConstructorFromBytes) {
  std::vector<unsigned char> initial_state(32, 0xAB);
  ASSERT_NO_THROW(
      ElementalCelularAutomata ca(initial_state, 256, 30));
  ElementalCelularAutomata ca(initial_state, 256, 30);
  ASSERT_NE(ca.get_cuda_state(), nullptr);
}

TEST_F(CellularAutomataTest, IterationChangesState) {
  std::vector<unsigned int> initial_state(8, 0x00000001);
  ElementalCelularAutomata ca(initial_state, 256, 30);

  std::vector<unsigned int> state_before(8);
  cudaMemcpy(state_before.data(), ca.get_cuda_state(), 8 * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  ca.iterate(5);

  std::vector<unsigned int> state_after(8);
  cudaMemcpy(state_after.data(), ca.get_cuda_state(), 8 * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  ASSERT_NE(state_before, state_after);
}

TEST_F(CellularAutomataTest, MultipleIterations) {
  ElementalCelularAutomata ca(512, 30);

  for (int i = 0; i < 10; ++i) {
    ASSERT_NO_THROW(ca.iterate(1));
  }
}

TEST_F(CellularAutomataTest, DifferentRules) {
  std::vector<int> rules = {30, 110, 150, 184, 225};
  for (int rule : rules) {
    ASSERT_NO_THROW(ElementalCelularAutomata ca(256, rule));
  }
}

TEST_F(CellularAutomataTest, BlockLevelIteration) {
  ElementalCelularAutomata ca(1024, 30);
  ASSERT_NO_THROW(ca.iterate_block_level(5));
}

TEST_F(CellularAutomataTest, VariousSizes) {
  std::vector<size_t> sizes = {64, 128, 256, 512, 1024};
  for (size_t size : sizes) {
    ASSERT_NO_THROW(ElementalCelularAutomata ca(size, 30));
  }
}

// ============================================================================
// TEST GROUP 3: Memory Allocation and Transfer Tests
// ============================================================================

class MemoryTest : public CipherTest {};

TEST_F(MemoryTest, AllocateSmallImage) {
  cv::Mat img = create_test_image(64, 64, 1);
  D_pointers d_ptrs;

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;

  ASSERT_NO_THROW(allocate_and_transfer_image(d_ptrs, img, params, false));
  ASSERT_NE(d_ptrs.d_image, nullptr);
}

TEST_F(MemoryTest, AllocateLargeImage) {
  cv::Mat img = create_test_image(256, 256, 1);
  D_pointers d_ptrs;

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 16;
  params.automata_steps = 15;
  params.transition_length = 8;
  params.chaos_parameter = 3.9f;

  ASSERT_NO_THROW(allocate_and_transfer_image(d_ptrs, img, params, false));
  ASSERT_NE(d_ptrs.d_image, nullptr);
}

TEST_F(MemoryTest, TransferBackAndCleanup) {
  cv::Mat img = create_test_image(64, 64, 1);
  cv::Mat img_result = img.clone();
  D_pointers d_ptrs;

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;

  ASSERT_NO_THROW(allocate_and_transfer_image(d_ptrs, img, params, false));
  ASSERT_NO_THROW(transfer_back_and_cleanup(d_ptrs, img_result));
}

// ============================================================================
// TEST GROUP 5: Image Encryption/Decryption Tests
// ============================================================================

class EncryptionTest : public CipherTest {};

TEST_F(EncryptionTest, EncryptGrayscaleSmall) {
  cv::Mat img = create_test_image(64, 64, 1);
  
  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;
  
  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(EncryptionTest, EncryptDecryptInvertible) {
  cv::Mat original = create_test_image(64, 64, 1);
  cv::Mat img = original.clone();

  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  // Encrypt
  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
  cv::Mat encrypted = img.clone();

  // Verify encryption changed the image
  ASSERT_NE(cv::countNonZero(original != encrypted), 0)
      << "Image should change after encryption";

  // Decrypt
  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, false));

  // Verify decryption restored original
  ASSERT_EQ(cv::countNonZero(original != img), 0)
      << "Image should be restored after decrypt";
}

TEST_F(EncryptionTest, DifferentPasswordsProduceDifferentCiphertexts) {
  cv::Mat img1 = create_test_image(64, 64, 1);
  cv::Mat img2 = img1.clone();

  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password1 =
      create_password_for_image(dims);
  std::vector<std::vector<unsigned char>> password2 =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  encrypt_image(img1, password1, dims, params, false, true);
  encrypt_image(img2, password2, dims, params, false, true);

  ASSERT_NE(cv::countNonZero(img1 != img2), 0)
      << "Different passwords should produce different ciphertexts";
}

TEST_F(EncryptionTest, VariousRoundCounts) {
  std::vector<size_t> rounds = {1, 2, 3, 4, 5};

  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  for (size_t r : rounds) {
    cv::Mat img = create_test_image(64, 64, 1);
    std::vector<std::vector<unsigned char>> password =
        create_password_for_image(dims);

    EncryptionParams params;
    params.rounds = r;
    params.block_size = 8;
    params.automata_steps = 10;
    params.transition_length = 5;
    params.chaos_parameter = 3.9f;
    params.image_hash = 0;

    ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true))
        << "Encryption failed with " << r << " rounds";
  }
}

TEST_F(EncryptionTest, VariousBlockSizes) {
  std::vector<size_t> block_sizes = {4, 8, 16, 32};

  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  for (size_t bs : block_sizes) {
    cv::Mat img = create_test_image(64, 64, 1);
    std::vector<std::vector<unsigned char>> password =
        create_password_for_image(dims);

    EncryptionParams params;
    params.rounds = 1;
    params.block_size = bs;
    params.automata_steps = 10;
    params.transition_length = 5;
    params.chaos_parameter = 3.9f;
    params.image_hash = 0;

    ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true))
        << "Encryption failed with block size " << bs;
  }
}

// ============================================================================
// TEST GROUP 6: Various Image Sizes
// ============================================================================

class ImageSizeTest : public CipherTest {};

TEST_F(ImageSizeTest, SmallImage32x32) {
  cv::Mat img = create_test_image(32, 32, 1);
  Image_dimensions dims;
  dims.rows = 32;
  dims.cols = 32;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(ImageSizeTest, MediumImage128x128) {
  cv::Mat img = create_test_image(128, 128, 1);
  Image_dimensions dims;
  dims.rows = 128;
  dims.cols = 128;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 2;
  params.block_size = 16;
  params.automata_steps = 15;
  params.transition_length = 8;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(ImageSizeTest, LargeImage256x256) {
  cv::Mat img = create_test_image(256, 256, 1);
  Image_dimensions dims;
  dims.rows = 256;
  dims.cols = 256;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 3;
  params.block_size = 16;
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(ImageSizeTest, NonSquareImage) {
  cv::Mat img = create_test_image(128, 64, 1);
  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 128;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

// ============================================================================
// TEST GROUP 7: Chaos Parameter Tests
// ============================================================================

class ChaosParameterTest : public CipherTest {};

TEST_F(ChaosParameterTest, ChaosParameterVariations) {
  std::vector<float> chaos_params = {3.6f, 3.7f, 3.8f, 3.9f, 4.0f};

  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  for (float r : chaos_params) {
    cv::Mat img = create_test_image(64, 64, 1);
    std::vector<std::vector<unsigned char>> password =
        create_password_for_image(dims);

    EncryptionParams params;
    params.rounds = 1;
    params.block_size = 8;
    params.automata_steps = 10;
    params.transition_length = 5;
    params.chaos_parameter = r;
    params.image_hash = 0;

    ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true))
        << "Encryption failed with chaos parameter " << r;
  }
}

// ============================================================================
// TEST GROUP 8: Edge Cases and Stress Tests
// ============================================================================

class EdgeCaseTest : public CipherTest {};

TEST_F(EdgeCaseTest, AllZeroImage) {
  cv::Mat img = cv::Mat::zeros(64, 64, CV_8U);
  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(EdgeCaseTest, AllMaxImage) {
  cv::Mat img = cv::Mat::ones(64, 64, CV_8U) * 255;
  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(EdgeCaseTest, AlternatingPatternImage) {
  cv::Mat img(64, 64, CV_8U);
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      img.at<unsigned char>(i, j) = ((i + j) % 2) * 255;
    }
  }

  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(EdgeCaseTest, GradientImage) {
  cv::Mat img(64, 64, CV_8U);
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      img.at<unsigned char>(i, j) = (i + j) % 256;
    }
  }

  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 10;
  params.transition_length = 5;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(EdgeCaseTest, MultipleEncryptRoundsDecrypt) {
  cv::Mat original = create_test_image(64, 64, 1);
  cv::Mat img = original.clone();

  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 5;
  params.block_size = 8;
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  encrypt_image(img, password, dims, params, false, true);
  ASSERT_NE(cv::countNonZero(original != img), 0);

  encrypt_image(img, password, dims, params, false, false);
  ASSERT_EQ(cv::countNonZero(original != img), 0)
      << "Multi-round encryption should be invertible";
}

TEST_F(EdgeCaseTest, LowAutomataSteps) {
  cv::Mat img = create_test_image(64, 64, 1);
  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 1;
  params.transition_length = 1;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(EdgeCaseTest, HighAutomataSteps) {
  cv::Mat img = create_test_image(64, 64, 1);
  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 8;
  params.automata_steps = 100;
  params.transition_length = 50;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

// ============================================================================
// TEST GROUP 9: Integration Tests
// ============================================================================

class IntegrationTest : public CipherTest {};

TEST_F(IntegrationTest, FullEncryptDecryptPipeline) {
  cv::Mat original = create_test_image(128, 128, 1);
  cv::Mat img = original.clone();

  Image_dimensions dims;
  dims.rows = 128;
  dims.cols = 128;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 3;
  params.block_size = 16;
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  encrypt_image(img, password, dims, params, false, true);
  encrypt_image(img, password, dims, params, false, false);

  ASSERT_EQ(cv::countNonZero(original != img), 0)
      << "Full pipeline should preserve image";
}

TEST_F(IntegrationTest, MultipleImagesConsistent) {
  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 2;
  params.block_size = 8;
  params.automata_steps = 15;
  params.transition_length = 8;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  cv::Mat img1 = create_test_image(64, 64, 1);
  cv::Mat img2 = img1.clone();

  encrypt_image(img1, password, dims, params, false, true);
  encrypt_image(img2, password, dims, params, false, true);

  ASSERT_EQ(cv::countNonZero(img1 != img2), 0)
      << "Same image with same password should produce same ciphertext";
}

TEST_F(IntegrationTest, DifferentPasswordsDifferentCiphertexts) {
  cv::Mat original = create_test_image(64, 64, 1);

  Image_dimensions dims;
  dims.rows = 64;
  dims.cols = 64;

  std::vector<std::vector<unsigned char>> password1 =
      create_password_for_image(dims);
  std::vector<std::vector<unsigned char>> password2 =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 2;
  params.block_size = 8;
  params.automata_steps = 15;
  params.transition_length = 8;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  cv::Mat encrypted1 = original.clone();
  cv::Mat encrypted2 = original.clone();

  encrypt_image(encrypted1, password1, dims, params, false, true);
  encrypt_image(encrypted2, password2, dims, params, false, true);

  ASSERT_NE(cv::countNonZero(encrypted1 != encrypted2), 0)
      << "Different passwords must produce different results";
}

// ============================================================================
// TEST GROUP 10: Large Block Sizes Tests
// ============================================================================

class LargeBlockSizeTest : public CipherTest {};

TEST_F(LargeBlockSizeTest, BlockSize32WithMediumImage) {
  cv::Mat img = create_test_image(128, 128, 1);
  Image_dimensions dims;
  dims.rows = 128;
  dims.cols = 128;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 2;
  params.block_size = 32;
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(LargeBlockSizeTest, BlockSize64WithLargeImage) {
  cv::Mat img = create_test_image(256, 256, 1);
  Image_dimensions dims;
  dims.rows = 256;
  dims.cols = 256;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 2;
  params.block_size = 64;
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(LargeBlockSizeTest, BlockSize128WithVeryLargeImage) {
  cv::Mat img = create_test_image(512, 512, 1);
  Image_dimensions dims;
  dims.rows = 512;
  dims.cols = 512;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 2;
  params.block_size = 128;
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(LargeBlockSizeTest, EncryptDecryptWithLargeBlockSize) {
  cv::Mat original = create_test_image(256, 256, 1);
  cv::Mat img = original.clone();

  Image_dimensions dims;
  dims.rows = 256;
  dims.cols = 256;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 2;
  params.block_size = 64;
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  encrypt_image(img, password, dims, params, false, true);
  ASSERT_NE(cv::countNonZero(original != img), 0);

  encrypt_image(img, password, dims, params, false, false);
  ASSERT_EQ(cv::countNonZero(original != img), 0)
      << "Large block size encryption should be invertible";
}

// ============================================================================
// TEST GROUP 11: Large Image Stress Tests
// ============================================================================

class LargeImageStressTest : public CipherTest {};

TEST_F(LargeImageStressTest, LargeRandom512x512Image) {
  cv::Mat img = create_test_image(512, 512, 1);
  Image_dimensions dims;
  dims.rows = 512;
  dims.cols = 512;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 2;
  params.block_size = 64;
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(LargeImageStressTest, LargeImage1024x1024) {
  cv::Mat img = create_test_image(1024, 1024, 1);
  Image_dimensions dims;
  dims.rows = 1024;
  dims.cols = 1024;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 1;
  params.block_size = 128;
  params.automata_steps = 15;
  params.transition_length = 8;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));
}

TEST_F(LargeImageStressTest, BlockSize32WithVaryingRounds) {
  Image_dimensions dims;
  dims.rows = 256;
  dims.cols = 256;

  for (size_t rounds = 1; rounds <= 4; ++rounds) {
    cv::Mat img = create_test_image(256, 256, 1);
    std::vector<std::vector<unsigned char>> password =
        create_password_for_image(dims);

    EncryptionParams params;
    params.rounds = rounds;
    params.block_size = 32;
    params.automata_steps = 20;
    params.transition_length = 10;
    params.chaos_parameter = 3.9f;
    params.image_hash = 0;

    ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true))
        << "Failed with block_size=32, rounds=" << rounds;
  }
}

TEST_F(LargeImageStressTest, BlockSize64AndLargeRounds) {
  cv::Mat original = create_test_image(256, 256, 1);
  cv::Mat img = original.clone();

  Image_dimensions dims;
  dims.rows = 256;
  dims.cols = 256;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 4;
  params.block_size = 64;
  params.automata_steps = 25;
  params.transition_length = 12;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  encrypt_image(img, password, dims, params, false, true);
  ASSERT_NE(cv::countNonZero(original != img), 0);

  encrypt_image(img, password, dims, params, false, false);
  ASSERT_EQ(cv::countNonZero(original != img), 0)
      << "Decryption failed with block_size=64, rounds=4";
}

// ============================================================================
// TEST GROUP 9: Scalability Tests
// ============================================================================

TEST_F(EdgeCaseTest, LargeImage1024x1024) {
  // Test with 1024x1024 image (approx Scale 4.0x from 256x256)
  // This aims to reproduce the metadata corruption crash seen in stats.py
  int size = 1024;
  cv::Mat img = create_test_image(size, size, 1);
  Image_dimensions dims;
  dims.rows = size;
  dims.cols = size;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 3;
  params.block_size = 16; // Using 16 (Safe for Large Images)
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  cv::Mat original = img.clone();
  
  // Encrypt
  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));

  // Decrypt
  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, false));
  
  // Verify equality
  int diff = cv::countNonZero(original != img);
  ASSERT_EQ(diff, 0) << "Decryption failed for 1024x1024 image. Diff pixels: " << diff;
}

// ============================================================================
// Main function and test runner configuration
// ============================================================================

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::cout << "=== Comprehensive Unit Test Suite for Image Cipher ===" << std::endl;
  std::cout << "Running tests for encryption functions and kernels..." << std::endl;
  std::cout << std::endl;

  return RUN_ALL_TESTS();
}
