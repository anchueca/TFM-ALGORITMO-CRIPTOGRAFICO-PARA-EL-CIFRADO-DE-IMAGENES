#ifndef STRUCT_CUH
#define STRUCT_CUH

/**
 * @brief Parameters used to control the encryption pipeline.
 *
 * - rounds: Number of encryption rounds.
 * - block_size: Size of square blocks used for block-phase permutations.
 * - precision_level: Precision parameter for automata/password derivation.
 * - automata_steps: Number of automata evolution steps used to generate
 * permutations.
 * - transition_length: Length of transition sequence used in flow permutations.
 */
struct EncryptionParams {
  size_t rounds;
  size_t block_size;
  size_t precision_level;
  size_t automata_steps;
  size_t transition_length;
};

struct Image_dimnesions {
  size_t cols;
  size_t rows;
};

struct D_pointers {
  unsigned char *d_image;
  unsigned char *d_image_out;
  unsigned char *d_flow;
  unsigned int *d_permutation_rows;
  unsigned int *d_permutation_cols;
  unsigned int *d_permutation_blocks;
};

#endif // STRUCT_CUH