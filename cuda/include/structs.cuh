#ifndef STRUCT_CUH
#define STRUCT_CUH

/**
 * @file structs.cuh
 * @brief Data structures used throughout the encryption system.
 */

/**
 * @brief Parameters used to control the encryption pipeline.
 *
 * This struct configures the behavior of the image encryption algorithm.
 * All parameters significantly affect both security and performance.
 *
 * @par Member Descriptions:
 * - @b rounds: Number of complete encryption cycles.
 *   - Each round applies: confusion (XOR with chaotic flow) + diffusion
 * (permutations)
 *   - More rounds = stronger avalanche effect, slower performance
 *
 * - @b block_size: Dimension of square blocks for block-level permutations.
 *   - Must divide image dimensions evenly (partial blocks handled
 * automatically)
 *
 * - @b automata_steps: Number of evolution steps for elementary cellular
 * automata.
 *   - More steps = more random permutations, higher initialization overhead
 *   - Used to generate row and column permutations
 *
 * - @b transition_length: Number of iterations for chaotic flow permutation
 * generation.
 *   - More transitions = better mixing of chaotic sequences
 *   - Used in block permutation generation
 *
 * - @b chaos_parameter: Parameter 'r' for the logistic map: x_{n+1} = r * x_n *
 * (1 - x_n)
 *   - Values outside chaotic regime will produce weak encryption!
 *   - Verify chaotic behavior using bifurcation diagrams (see
 * python/bifurcacion.py)
 *
 * @par Example Usage:
 * @code
 * EncryptionParams params;
 * params.rounds = 3;
 * params.block_size = 8;
 * params.precision_level = 4;
 * params.automata_steps = 20;
 * params.transition_length = 10;
 * params.chaos_parameter = 3.9;
 * @endcode
 *
 * @note All encryption and decryption operations must use identical parameters!
 *       Parameter mismatch will result in incorrect decryption.
 */
#ifdef USE_DOUBLE_PRECISION
using Real = double;
#else
using Real = float;
#endif

struct EncryptionParams {
  size_t rounds;
  size_t block_size;
  size_t automata_steps;
  size_t transition_length;
  Real chaos_parameter;
  unsigned short image_hash;
};

/**
 * @brief Image dimension information.
 *
 * Stores the dimensions of the image to be encrypted/decrypted.
 * Used throughout the pipeline to configure kernel launch parameters.
 *
 * @note These dimensions refer to the processed image size (after RGB
 * unstacking for color images). For a color image, cols is multiplied by 3.
 */
struct Image_dimensions {
  size_t cols; ///< Number of columns (width in pixels, or width*3 for RGB)
  size_t rows; ///< Number of rows (height in pixels)
};

/**
 * @brief Device (GPU) memory pointers for the encryption pipeline.
 *
 * This struct centralizes all GPU memory allocations required during
 * encryption/decryption. All pointers refer to device memory and must
 * be accessed only from GPU kernels or via CUDA memory operations.
 *
 * @par Memory Management:
 * - Pointers are allocated via cudaMalloc() before use
 * - Must be freed via cudaFree() after processing
 * - Double buffering (d_image, d_image_out) enables zero-copy swapping
 *
 * @par Pointer Descriptions:
 * - @b d_image: Current image buffer (input for current operation)
 * - @b d_image_out: Output image buffer (result of current operation)
 *   These are swapped between operations to avoid copying.
 *
 * - @b d_flow: Chaotic flow sequence used for confusion (XOR) operations
 *
 * - @b d_seeds: Random seeds derived from password, used to initialize
 *   chaotic sequences and automata
 *
 * - @b d_permutation_rows: Row permutation indices (forward)
 * - @b d_permutation_cols: Column permutation indices (forward)
 * - @b d_permutation_blocks: Block permutation indices (forward)
 *
 * - @b d_permutation_rows_inverse: Inverse row permutations
 * - @b d_permutation_cols_inverse: Inverse column permutations
 * - @b d_permutation_blocks_inverse: Inverse block permutations
 *
 * @note Inverse permutations are pre-computed during initialization
 */
struct D_pointers {
  unsigned char *d_image = nullptr;     ///< Current image data on device
  unsigned char *d_image_out = nullptr; ///< Output buffer for operations
  unsigned char *d_flow = nullptr; ///< Chaotic flow sequence (used for XOR)
  Real *d_seeds = nullptr; ///< Random seeds for initialization (used in flow)
                           ///< includes extra seeds for block permutations
  unsigned int *d_permutation_blocks = nullptr; ///< Forward block permutation
  unsigned int *d_permutation_blocks_inverse =
      nullptr; ///< Inverse block permutation
  
  // Unified permutation vector (P) and its inverse (P^-1)
  // Used for both rows and columns (Row=P, Col=P^-1)
  unsigned int *d_permutation_vector = nullptr; 
  unsigned int *d_permutation_vector_inverse = nullptr;
  Real *d_chaotic_values_for_permutation =
      nullptr;                    // For block permutation generation
  unsigned int *d_automata_state = nullptr; // For automata iteration in flow generation
  unsigned short *d_image_automata_state = nullptr; // Automata state for extra seeds.
};

#endif // STRUCT_CUH