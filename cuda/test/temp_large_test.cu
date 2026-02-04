TEST_F(EdgeCaseTest, LargeImage1024x1024) {
  // Test with 1024x1024 image (approx Scale 4.0x from 256x256)
  int size = 1024;
  cv::Mat img = create_test_image(size, size, 1);
  Image_dimensions dims;
  dims.rows = size;
  dims.cols = size;

  std::vector<std::vector<unsigned char>> password =
      create_password_for_image(dims);

  EncryptionParams params;
  params.rounds = 3;
  params.block_size = 16;
  params.automata_steps = 20;
  params.transition_length = 10;
  params.chaos_parameter = 3.9f;
  params.image_hash = 0;

  // Run full pipeline: Encrypt -> Decrypt -> Compare
  // Note: encrypt_image modifies 'img' in place to ciphertext
  cv::Mat original = img.clone();
  
  // Encrypt (in-place)
  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, true));

  // Decrypt (in-place)
  ASSERT_NO_THROW(encrypt_image(img, password, dims, params, false, false));
  
  // Verify equality with original
  int diff = cv::countNonZero(original != img);
  ASSERT_EQ(diff, 0) << "Decryption failed for large image. Diff pixels: " << diff;
}
