/**
 * @file kernels_aux.cu
 * @brief Implementations of small CUDA kernels used for image tiling and
 * permutation helpers.
 */

#include "../include/kernels_aux.cuh"


__global__ void invert_permutations_kernel(unsigned int *permutations,
                                           unsigned int *inverses,
                                           size_t block_length,
                                           size_t num_blocks) {
  // Each block handles a single permutation inversion. Grid-stride loop covers
  // all elements.
  int permutation_id = blockIdx.x;
  int thread_id_in_block = threadIdx.x;
  int threads_per_block = blockDim.x;

  for (int i = thread_id_in_block; i < block_length; i += threads_per_block) {
    size_t idx_in = permutation_id * block_length + i;
    unsigned int new_pos = permutations[idx_in];
    size_t idx_out = permutation_id * block_length + new_pos;
    inverses[idx_out] = i;
  }
}

__global__ void image_xor(unsigned char *keystream, unsigned char *image,
                          Image_dimnesions img_dimensions) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= img_dimensions.cols || y >= img_dimensions.rows) {
      return;
  }

  int idx = y * img_dimensions.cols + x;

  image[idx] ^= keystream[idx];
}

__device__ unsigned char convertToBitStream(double value){

  unsigned long long raw_bits = __double_as_longlong(value);

  //I get the lsb bits of the antissa
  unsigned int lsb_mantissa = (unsigned int)raw_bits;
  
  unsigned char b1 = (lsb_mantissa >> 24) & 0xFF;
  unsigned char b2 = (lsb_mantissa >> 16) & 0xFF;
  unsigned char b3 = (lsb_mantissa >> 8)  & 0xFF;
  unsigned char b4 = lsb_mantissa & 0xFF;

  unsigned char keystream_byte = b1 ^ b2 ^ b3 ^ b4;

  return  keystream_byte;
}

__device__ unsigned char convertToBitStream(float value){

  unsigned int raw_bits = __float_as_uint(value);

  unsigned int mantissa = raw_bits & 0xFFFF;
  
  unsigned char b1 = (mantissa >> 8) & 0xFF; // Bits 15-8
  unsigned char b2 = mantissa & 0xFF;        // Bits 7-0

  unsigned char keystream_byte = b1 ^ b2;

  return  keystream_byte;
}