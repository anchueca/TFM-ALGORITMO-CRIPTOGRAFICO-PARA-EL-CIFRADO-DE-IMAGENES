/**
 * @file visual_demo.cu
 * @brief Visual tests of encryption phases - padding, obfuscation, permutations
 *
 * Tests individual encryption phases with varying image sizes and parameters:
 * - Padding visualization
 * - Column permutation effects
 * - Block permutation with different block sizes
 * - Obfuscation/diffusion visualization
 * - Multiple image sizes (standard, non-standard, edge cases)
 */

#include <cmath>
#include <cstdlib>
#define rsqrt rsqrt_broken_glibc
#define rsqrtf rsqrtf_broken_glibc
#include <cmath>
#undef rsqrt
#undef rsqrtf
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "../include/automata.cuh"
#include "../include/encryption.cuh"
#include "../include/encryption_aux.cuh"
#include "../include/kernels.cuh"
#include "../include/structs.cuh"

// ============================================================================
// Helper: Load or generate test image
// ============================================================================
cv::Mat load_test_image(int width, int height, const char *name) {
  cv::Mat img(height, width, CV_8U);

  // Try to load from repository
  std::string image_paths[] = {
      "../repositorio/set1/circles.tif", "../repositorio/set1/crosses.tif",
      "../repositorio/set1/squares.tif", "../repositorio/set1/text.tif"};

  for (const auto &path : image_paths) {
    cv::Mat temp = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (!temp.empty()) {
      cv::resize(temp, img, cv::Size(width, height));
      return img;
    }
  }

  // Fallback: generate pattern
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int checker = 16;
      if (((i / checker) + (j / checker)) % 2 == 0) {
        img.at<unsigned char>(i, j) = 255;
      } else {
        img.at<unsigned char>(i, j) = 0;
      }
    }
  }
  return img;
}

// ============================================================================
// Helper: Save or Show image
// ============================================================================
void save_or_show(const std::string &title, const cv::Mat &img) {
  if (img.empty())
    return;

  const char *display = std::getenv("DISPLAY");
  if (display == nullptr || std::string(display).empty()) {
    // Headless mode: save to file
    std::string filename = title;
    // Clean filename: replace spaces and special chars
    for (char &c : filename) {
      if (c == ' ' || c == ':' || c == '/' || c == '(' || c == ')' || c == '=')
        c = '_';
    }
    std::string path = "visual_results/" + filename + ".png";
    cv::imwrite(path, img);
    std::cout << "  [Saved to " << path << "]" << std::endl;
  } else {
    // GUI mode: show window
    cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
    cv::imshow(title, img);
    cv::waitKey(500); // Brief pause to process events
  }
}

void wait_if_gui() {
  const char *display = std::getenv("DISPLAY");
  if (display != nullptr && !std::string(display).empty()) {
    std::cout << "\nPress any key in image window to continue..." << std::endl;
    cv::waitKey(0);
  }
  cv::destroyAllWindows();
}

// ============================================================================
// TEST 1: Padding Visualization
// ============================================================================
void test_padding() {
  std::cout << "\n" << std::string(70, '=') << std::endl;
  std::cout << "TEST 1: PADDING VISUALIZATION" << std::endl;
  std::cout << std::string(70, '=') << std::endl;

  // Test different odd/unusual image sizes
  int sizes[] = {137, 213, 255, 100, 513};

  for (int size : sizes) {
    std::cout << "\n► Testing size: " << size << "x" << size << std::endl;

    cv::Mat original = load_test_image(size, size, "original");
    cv::Mat padded = original.clone();

    // Calculate padding needed to make it 128-multiple
    int block_128 = ((size + 127) / 128) * 128;
    int pad_rows = block_128 - size;
    int pad_cols = block_128 - size;

    std::cout << "  Original: " << size << "x" << size << std::endl;
    std::cout << "  Padded to: " << block_128 << "x" << block_128
              << " (pad: " << pad_rows << "x" << pad_cols << ")" << std::endl;

    if (pad_rows > 0 || pad_cols > 0) {
      cv::copyMakeBorder(padded, padded, 0, pad_rows, 0, pad_cols,
                         cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    // Display side-by-side
    cv::Mat display;
    cv::Mat original_for_display = original.clone();
    if (pad_rows > 0 || pad_cols > 0) {
      cv::copyMakeBorder(original_for_display, original_for_display, 0,
                         pad_rows, 0, pad_cols, cv::BORDER_CONSTANT,
                         cv::Scalar(128)); // Light gray for visibility
    }
    cv::hconcat(original_for_display, padded, display);

    std::string title = "Padding_" + std::to_string(size);
    save_or_show(title, display);
  }

  std::cout << "\n✓ Padding tests complete." << std::endl;
  wait_if_gui();
}

// Helper for test identity permutations
unsigned int *get_identity_gpu(size_t n) {
  std::vector<unsigned int> h_id(n);
  for (size_t i = 0; i < n; ++i)
    h_id[i] = i;
  unsigned int *d_id;
  cudaMalloc(&d_id, n * sizeof(unsigned int));
  cudaMemcpy(d_id, h_id.data(), n * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  return d_id;
}

// ============================================================================
// TEST 2: ACTUAL Row and Column Permutations
// ============================================================================
void test_actual_permutations() {
  std::cout << "\n" << std::string(70, '=') << std::endl;
  std::cout << "TEST 2: ACTUAL ROW AND COLUMN PERMUTATIONS (via Fused Kernel)"
            << std::endl;
  std::cout << std::string(70, '=') << std::endl;

  int size = 256;
  cv::Mat original = load_test_image(size, size, "original");

  // Setup GPU memory
  unsigned char *d_img, *d_img_out;
  cudaMalloc(&d_img, size * size);
  cudaMalloc(&d_img_out, size * size);
  cudaMemcpy(d_img, original.data, size * size, cudaMemcpyHostToDevice);

  // Setup Password (2 bytes per index for the CA)
  std::vector<unsigned char> pass_ca(size * 2, 0xAB);
  ElementalCelularAutomata ca(pass_ca, size * 16, 30);

  // Generate Permutations
  unsigned int *d_perm = generate_automata_permutations(&ca, 20, size, false);

  Image_dimensions dims = {(size_t)size, (size_t)size};
  unsigned int *d_id = get_identity_gpu(size);
  unsigned int *d_id_blocks = get_identity_gpu(64); // 8x8 blocks

  // --- Test Columns ---
  std::cout << "► Testing actual column permutation (isolated)..." << std::endl;
  // Fused: perm_rows=id, perm_cols=perm, blocks=id
  fused_permutation_xor(d_img, d_img_out, nullptr, d_id, d_perm, d_id_blocks,
                        d_id_blocks, dims, 8, false, false);
  cudaDeviceSynchronize();

  cv::Mat res_cols(size, size, CV_8U);
  cudaMemcpy(res_cols.data, d_img_out, size * size, cudaMemcpyDeviceToHost);
  cv::Mat disp_cols;
  cv::hconcat(original, res_cols, disp_cols);
  save_or_show("Permutation_Cols_Isolated", disp_cols);

  // --- Test Rows ---
  std::cout << "► Testing actual row permutation (isolated)..." << std::endl;
  // Fused: perm_rows=perm, perm_cols=id, blocks=id
  fused_permutation_xor(d_img, d_img_out, nullptr, d_perm, d_id, d_id_blocks,
                        d_id_blocks, dims, 8, false, false);
  cudaDeviceSynchronize();

  cv::Mat res_rows(size, size, CV_8U);
  cudaMemcpy(res_rows.data, d_img_out, size * size, cudaMemcpyDeviceToHost);
  cv::Mat disp_rows;
  cv::hconcat(original, res_rows, disp_rows);
  save_or_show("Permutation_Rows_Isolated", disp_rows);

  cudaFree(d_id);
  cudaFree(d_id_blocks);
  cudaFree(d_img);
  cudaFree(d_img_out);
  cudaFree(d_perm);

  std::cout << "\n✓ Row/Col permutation tests complete." << std::endl;
  wait_if_gui();
}

// ============================================================================
// TEST 3: ACTUAL Block Permutation
// ============================================================================
void test_actual_block_permutation() {
  std::cout << "\n" << std::string(70, '=') << std::endl;
  std::cout << "TEST 3: ACTUAL BLOCK PERMUTATION (via Fused Kernel)"
            << std::endl;
  std::cout << std::string(70, '=') << std::endl;

  int size = 256;
  int block_size = 8;
  cv::Mat original = load_test_image(size, size, "original");

  D_pointers d_ptrs;
  memset(&d_ptrs, 0, sizeof(D_pointers));

  Image_dimensions dims = {(size_t)size, (size_t)size};
  EncryptionParams params;
  params.block_size = block_size;
  params.chaos_parameter = 3.9;
  params.transition_length = 50;

  cudaMalloc(&d_ptrs.d_image, size * size);
  cudaMalloc(&d_ptrs.d_image_out, size * size);
  cudaMemcpy(d_ptrs.d_image, original.data, size * size,
             cudaMemcpyHostToDevice);

  // Need chaotic values for block permutation
  cudaMalloc(&d_ptrs.d_chaotic_values_for_permutation,
             block_size * block_size * sizeof(Real));

  // Fill chaotic values with some pattern
  std::vector<Real> h_chaotic(block_size * block_size);
  for (int i = 0; i < block_size * block_size; ++i)
    h_chaotic[i] = (Real)sin(i * 0.7 + 0.1);
  cudaMemcpy(d_ptrs.d_chaotic_values_for_permutation, h_chaotic.data(),
             block_size * block_size * sizeof(Real), cudaMemcpyHostToDevice);

  std::cout
      << "► Generating and applying actual block permutation (isolated)..."
      << std::endl;
  generate_permutation_block(d_ptrs, dims, params);

  unsigned int *d_id = get_identity_gpu(size);

  // Use fused for block permutation only
  fused_permutation_xor(d_ptrs.d_image, d_ptrs.d_image_out, nullptr, d_id, d_id,
                        d_ptrs.d_permutation_blocks,
                        d_ptrs.d_permutation_blocks_inverse, dims, block_size,
                        false, false);
  cudaDeviceSynchronize();

  cv::Mat result(size, size, CV_8U);
  cudaMemcpy(result.data, d_ptrs.d_image_out, size * size,
             cudaMemcpyDeviceToHost);

  cv::Mat display;
  cv::hconcat(original, result, display);
  save_or_show("Block_Permutation_Isolated", display);

  // Cleanup
  cudaFree(d_id);
  cudaFree(d_ptrs.d_image);
  cudaFree(d_ptrs.d_image_out);
  cudaFree(d_ptrs.d_permutation_blocks);
  cudaFree(d_ptrs.d_permutation_blocks_inverse);
  cudaFree(d_ptrs.d_chaotic_values_for_permutation);

  std::cout << "\n✓ Block permutation test complete." << std::endl;
  wait_if_gui();
}

// ============================================================================
// TEST 4: ACTUAL Keystream and Diffusion (Fused)
// ============================================================================
void test_keystream_diffusion() {
  std::cout << "\n" << std::string(70, '=') << std::endl;
  std::cout << "TEST 4: ACTUAL KEYSTREAM AND DIFFUSION (Fused)" << std::endl;
  std::cout << std::string(70, '=') << std::endl;

  int size = 256;
  cv::Mat original = load_test_image(size, size, "original");

  D_pointers d_ptrs;
  memset(&d_ptrs, 0, sizeof(D_pointers));

  Image_dimensions dims = {(size_t)size, (size_t)size};
  EncryptionParams params;
  params.chaos_parameter = 3.99;
  params.transition_length = 100;
  params.block_size = 8;
  params.rounds = 1;
  params.image_hash = 0x1234;

  cudaMalloc(&d_ptrs.d_image, size * size);
  cudaMalloc(&d_ptrs.d_image_out, size * size);
  cudaMemcpy(d_ptrs.d_image, original.data, size * size,
             cudaMemcpyHostToDevice);
  cudaMalloc(&d_ptrs.d_flow, size * size);

  // Setup seeds from password
  std::vector<unsigned char> password_bytes(dims.cols * 2 * 4, 0x55);
  convert_bits_to_real(password_bytes, &d_ptrs.d_seeds);

  // Need CA state for flow generation (2 bytes per column)
  std::vector<unsigned char> pass_ca(dims.cols * 2, 0x12);
  ElementalCelularAutomata ca(pass_ca, dims.cols * 16, 30);
  size_t state_size = ca.get_size_in_bytes();
  cudaMalloc(&d_ptrs.d_automata_state, state_size);
  cudaMemcpy(d_ptrs.d_automata_state, ca.get_cuda_state(), state_size,
             cudaMemcpyDeviceToDevice);

  std::cout << "► Generating actual keystream (CML)..." << std::endl;
  generate_flow_stream_parallel(d_ptrs, dims, params);

  cv::Mat keystream(size, size, CV_8U);
  cudaMemcpy(keystream.data, d_ptrs.d_flow, size * size,
             cudaMemcpyDeviceToHost);
  save_or_show("Keystream_Noise", keystream);

  std::cout << "► Applying fused permutation and diffusion (XOR)..."
            << std::endl;

  unsigned int *d_id = get_identity_gpu(size);
  unsigned int *d_id_blocks = get_identity_gpu(64);

  // Apply fused: I_out = Perm(I_in) ^ Flow
  fused_permutation_xor(d_ptrs.d_image, d_ptrs.d_image_out, d_ptrs.d_flow, d_id,
                        d_id, d_id_blocks, d_id_blocks, dims, 8, true, false);
  cudaDeviceSynchronize();

  cv::Mat result(size, size, CV_8U);
  cudaMemcpy(result.data, d_ptrs.d_image_out, size * size,
             cudaMemcpyDeviceToHost);
  save_or_show("Fused_XOR_Diffusion_Result", result);

  cudaFree(d_id);
  cudaFree(d_id_blocks);

  // Cleanup
  cudaFree(d_ptrs.d_image);
  cudaFree(d_ptrs.d_image_out);
  cudaFree(d_ptrs.d_flow);
  cudaFree(d_ptrs.d_seeds);
  cudaFree(d_ptrs.d_automata_state);
  cudaFree(d_ptrs.d_image_automata_state);
  if (d_ptrs.d_chaotic_values_for_permutation)
    cudaFree(d_ptrs.d_chaotic_values_for_permutation);

  std::cout << "\n✓ Keystream/Diffusion tests complete." << std::endl;
  wait_if_gui();
}

// ============================================================================
// TEST 5: Combined Phases Test
// ============================================================================
void test_combined_phases() {
  std::cout << "\n" << std::string(70, '═') << std::endl;
  std::cout << "TEST 5: COMBINED PHASE PROGRESSION" << std::endl;
  std::cout << std::string(70, '═') << std::endl;

  cv::Mat img = load_test_image(256, 256, "test");

  std::cout << "\nShowing encryption phase progression (left=original, "
               "right=encrypted)"
            << std::endl;

  cv::Mat encrypted = img.clone();

  // Simulate phase 1: permute columns
  for (int col = 0; col < 256; ++col) {
    int shift = (col * 7) % 256;
    cv::Mat temp = encrypted.col(col).clone();
    temp.copyTo(encrypted.col((col + shift) % 256));
  }

  // Simulate phase 2: permute rows
  for (int row = 0; row < 256; ++row) {
    int shift = (row * 13) % 256;
    cv::Mat temp = encrypted.row(row).clone();
    temp.copyTo(encrypted.row((row + shift) % 256));
  }

  // Simulate phase 3: block permutation
  for (int bi = 0; bi < 16; ++bi) {
    for (int bj = 0; bj < 16; ++bj) {
      int new_i = (bi * 5 + 3) % 16;
      int new_j = (bj * 7 + 2) % 16;

      cv::Mat block = encrypted(cv::Rect(bj * 16, bi * 16, 16, 16)).clone();

      block.copyTo(encrypted(cv::Rect(new_j * 16, new_i * 16, 16, 16)));
    }
  }

  // Simulate phase 4: XOR diffusion
  for (int i = 0; i < 256; ++i) {
    for (int j = 0; j < 256; ++j) {
      unsigned int seed = (i * 98765 + j * 43210) ^ 0xCAFEBABE;
      seed ^= (seed << 13);
      seed ^= (seed >> 17);
      encrypted.at<unsigned char>(i, j) ^= (seed & 0xFF);
    }
  }

  cv::Mat display;
  cv::hconcat(img, encrypted, display);

  save_or_show("Complete_Encryption_Flow", display);

  std::cout << "✓ Phase progression complete" << std::endl;
  wait_if_gui();
}

// ============================================================================
// TEST 6: Edge Cases and Unusual Sizes
// ============================================================================
void test_edge_cases() {
  std::cout << "\n" << std::string(70, '═') << std::endl;
  std::cout << "TEST 6: EDGE CASES - UNUSUAL IMAGE SIZES" << std::endl;
  std::cout << std::string(70, '═') << std::endl;

  int edge_sizes[] = {1, 7, 17, 31, 67, 127, 131, 259, 499, 1024};

  for (int size : edge_sizes) {
    std::cout << "\n► Size: " << size << "x" << size << " - ";

    // Calculate padding
    int block_128 = ((size + 127) / 128) * 128;

    if (block_128 % 128 == 0) {
      std::cout << "Valid (pads to " << block_128 << ")" << std::endl;
    } else {
      std::cout << "WARNING: Unusual padding required!" << std::endl;
    }

    cv::Mat img = load_test_image(size, size, "edge");
    cv::Mat padded = img.clone();

    if (padded.rows < block_128 || padded.cols < block_128) {
      cv::copyMakeBorder(padded, padded, 0, block_128 - size, 0,
                         block_128 - size, cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    cv::Mat display;
    cv::Mat img_for_display = img.clone();
    if (block_128 - size > 0) {
      cv::copyMakeBorder(img_for_display, img_for_display, 0, block_128 - size,
                         0, block_128 - size, cv::BORDER_CONSTANT,
                         cv::Scalar(128));
    }
    cv::hconcat(img_for_display, padded, display);

    std::string title = "EdgeCase_" + std::to_string(size);
    save_or_show(title, display);
  }

  std::cout << "\n✓ Edge case tests complete." << std::endl;
  wait_if_gui();
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv) {
  std::cout << "\n" << std::string(70, '═') << std::endl;
  std::cout << "IMAGE CIPHER - VISUAL PHASE TESTS" << std::endl;
  std::cout << "Testing individual encryption phases with various parameters"
            << std::endl;
  std::cout << std::string(70, '═') << std::endl;

  cudaSetDevice(0);
  cudaFree(0);

  test_padding();
  test_actual_permutations();
  test_actual_block_permutation();
  test_keystream_diffusion();
  test_combined_phases();
  test_edge_cases();

  std::cout << "\n" << std::string(70, '=') << std::endl;
  std::cout << std::string(70, '=') << std::endl;
  std::cout << std::string(70, '=') << std::endl << std::endl;

  return 0;
}
