#ifndef AUTOMATAKERNEL_CUH
#define AUTOMATAKERNEL_CUH

#include <cuda_runtime.h>

// --- CUDA Kernel Declaration ---
__global__ void evolve_shared(const unsigned int *current_state,
                              unsigned int *next_state, int rule, int size);

/**
 * @brief Block-level evolution kernel where each block operates independently.
 *
 * This kernel performs multiple iterations of cellular automaton evolution
 * with block-level boundary conditions. Each thread block operates as an
 * independent automaton where edge cells wrap around within the block.
 *
 * @param state Device pointer to the state buffer (input and output)
 * @param temp_state Device pointer to temporary state buffer for ping-pong
 * @param rule Rule number (0-255) defining the elementary automaton
 * @param num_steps Number of iterations to perform inside the kernel
 */
__global__ void evolve_block_level(unsigned int *state,
                                   unsigned int *temp_state, int rule,
                                   int num_steps);

/**
 * @brief Kernel for 16-bit isolated block evolution.
 *
 * Each thread handles a 16-bit block exclusively using registers.
 * No shared memory or global synchronization required.
 *
 * @param state Device pointer to state (viewed as unsigned short*)
 * @param rule Rule number (0-255)
 * @param num_steps Number of iterations
 */
__global__ void evolve_16bit_isolated(unsigned short *state, int rule,
                                      int num_steps);

#endif // AUTOMATAKERNEL_CUH