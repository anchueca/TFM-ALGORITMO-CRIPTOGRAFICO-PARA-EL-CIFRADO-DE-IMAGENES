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
 * @brief 16-bit isolated CA evolution helper.
 *
 * This variant takes the 16-bit state by value and returns the evolved
 * state. Passing by value avoids creating a pointer to a local variable
 * (which would force spills to local memory) and keeps the value in
 * registers, significantly reducing cost when called in a hot path.
 *
 * @param state 16-bit state value
 * @param rule Rule number (0-255)
 * @param num_steps Number of iterations
 * @return evolved 16-bit state
 */
static __device__ __forceinline__ unsigned short evolve_16bit_isolated(unsigned short state, int rule, int num_steps) {
    unsigned int current = state;

    for (int iter = 0; iter < num_steps; iter++) {
        unsigned int L = ((current >> 1) | (current << 15)) & 0xFFFF;
        unsigned int R = ((current << 1) | (current >> 15)) & 0xFFFF;
        unsigned int C = current & 0xFFFF;

        unsigned int next = 0;
        for (int p = 0; p < 8; p++) {
            if ((rule >> p) & 1) {
                unsigned int term = 0xFFFF;
                term &= (p & 4) ? L : ~L;
                term &= (p & 2) ? C : ~C;
                term &= (p & 1) ? R : ~R;
                next |= term;
            }
        }
        current = next & 0xFFFF;
    }
    return (unsigned short)current;
}

/**
 * @brief Optimized 16-bit isolated CA evolution for a single iteration of Rule 30.
 *
 * Computes Rule 30 directly using bitwise operations: L ^ (C | R)
 *
 * @param state 16-bit state value
 * @return evolved 16-bit state
 */
static __device__ __forceinline__ unsigned short evolve_16bit_isolated_rule30_1iter(unsigned short state) {
    unsigned int current = state;
    unsigned int L = ((current >> 1) | (current << 15)) & 0xFFFF;
    unsigned int R = ((current << 1) | (current >> 15)) & 0xFFFF;
    return (unsigned short)((L ^ (current | R)) & 0xFFFF);
}

#endif // AUTOMATAKERNEL_CUH