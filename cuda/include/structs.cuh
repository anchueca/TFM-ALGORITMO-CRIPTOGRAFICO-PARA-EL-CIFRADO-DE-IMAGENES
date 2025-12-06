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
 *   - Valid range: 1-10
 *   - Typical values: 1-5
 *   - Each round applies: confusion (XOR with chaotic flow) + diffusion
 * (permutations)
 *   - More rounds = stronger avalanche effect, slower performance
 *   - Recommended: 3 for balanced security/performance
 *
 * - @b block_size: Dimension of square blocks for block-level permutations.
 *   - Valid range: 4-64 (must be power of 2 for best performance)
 *   - Typical values: 8, 16, 32
 *   - Smaller blocks = finer-grained diffusion, slower performance
 *   - Larger blocks = coarser diffusion, faster performance
 *   - Must divide image dimensions evenly (partial blocks handled
 * automatically)
 *   - Recommended: 8-16 for most images
 *
 * - @b precision_level: Precision parameter for key derivation and automata
 * initialization.
 *   - Valid range: 1-16
 *   - Typical values: 2, 4, 8
 *   - Higher precision = more secure key derivation
 *   - Affects the granularity of password-to-key transformation
 *   - Recommended: 4 for balanced security
 *
 * - @b automata_steps: Number of evolution steps for elementary cellular
 * automata.
 *   - Valid range: 1-1000
 *   - Typical values: 20-100
 *   - More steps = more random permutations, higher initialization overhead
 *   - Used to generate row and column permutations
 *   - Recommended: 20-50 for most cases
 *
 * - @b transition_length: Number of iterations for chaotic flow permutation
 * generation.
 *   - Valid range: 1-100
 *   - Typical values: 10-50
 *   - More transitions = better mixing of chaotic sequences
 *   - Used in block permutation generation
 *   - Recommended: 10-20 for balanced performance
 *
 * - @b chaos_parameter: Parameter 'r' for the logistic map: x_{n+1} = r * x_n *
 * (1 - x_n)
 *   - Valid range: 0.0 - 4.0 (must be in chaotic regime: 3.57 < r <= 4.0)
 *   - Typical values: 3.9, 3.999
 *   - Values outside chaotic regime will produce weak encryption!
 *   - Use 3.9 for good balance, 3.999 for maximum chaos
 *   - Verify chaotic behavior using bifurcation diagrams (see
 * python/bifurcacion.py)
 *   - Recommended: 3.9 or 3.999
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
 *
 * @warning The chaos_parameter MUST be in the chaotic regime (3.57, 4.0].
 *          Values outside this range will severely weaken security.
 */
#ifdef USE_DOUBLE_PRECISION
using Real = double;
#else
using Real = float;
#endif

// Include standard libs that might be needed for types if not already
#include <cmath>

struct EncryptionParams {
  size_t rounds;
  size_t block_size;
  size_t precision_level;
  size_t automata_steps;
  size_t transition_length;
  Real chaos_parameter;
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
 * - @b d_permutation_rows_inverse: Inverse row permutations (for decryption)
 * - @b d_permutation_cols_inverse: Inverse column permutations (for decryption)
 * - @b d_permutation_blocks_inverse: Inverse block permutations (for
 * decryption)
 *
 * @note Inverse permutations are pre-computed during initialization to
 *       accelerate decryption.
 */
struct D_pointers {
  unsigned char *d_image;                   ///< Current image data on device
  unsigned char *d_image_out;               ///< Output buffer for operations
  unsigned char *d_flow;                    ///< Chaotic flow sequence
  Real *d_seeds;                            ///< Random seeds for initialization
  unsigned int *d_permutation_rows;         ///< Forward row permutation
  unsigned int *d_permutation_cols;         ///< Forward column permutation
  unsigned int *d_permutation_blocks;       ///< Forward block permutation
  unsigned int *d_permutation_rows_inverse; ///< Inverse row permutation
  unsigned int *d_permutation_cols_inverse; ///< Inverse column permutation
  unsigned int *d_permutation_blocks_inverse; ///< Inverse block permutation
};

#endif // STRUCT_CUH