#include "../include/automataKernel.cuh"

// --- CUDA Kernel ---

/**
 * @brief Kernel that evolves the automaton using shared memory for neighbor
 * access.
 *
 * The kernel reads packed unsigned ints containing 32 cells each into shared
 * memory with left/right halo words to allow reading neighbor cells without
 * additional global loads. Each thread operates on a single cell (bit) and
 * writes the resulting alive bit into the next_state using atomic operations.
 *
 * @param current_state Device pointer to the packed current state.
 * @param next_state Device pointer to the packed next state (must be zeroed
 * prior to launch).
 * @param rule Rule number (0-255) defining the elementary automaton.
 * @param size Total number of cells in the automaton.
 */
__global__ void evolve_shared(const unsigned int *current_state,
                              unsigned int *next_state, int rule, int size) {
  // Shared memory is declared externally; its size is passed at launch
  extern __shared__ unsigned int s_data[];

  int tid = threadIdx.x;
  int block_offset = blockIdx.x * blockDim.x;
  int idx = block_offset + tid;

  // --- 1. Load data from global to shared memory (with halos) ---
  int uints_per_block = blockDim.x / 32;
  int shared_mem_idx = tid / 32;
  int total_uints = (size + 31) / 32;

  // Each warp loads one unsigned int from the global state to the center of
  // shared memory
  if (shared_mem_idx < uints_per_block) {
    int global_read_idx = (block_offset / 32) + shared_mem_idx;
    if (global_read_idx < total_uints) {
      s_data[shared_mem_idx + 1] = current_state[global_read_idx];
    }
  }

  // The first thread of the block loads the left halo (with periodic boundary
  // condition)
  if (tid == 0) {
    int global_uint_idx = block_offset / 32;
    int left_halo_idx =
        (global_uint_idx == 0) ? total_uints - 1 : global_uint_idx - 1;
    s_data[0] = current_state[left_halo_idx];
  }

  // The last thread of the block loads the right halo
  if (tid == blockDim.x - 1) {
    int global_uint_idx = (block_offset / 32) + uints_per_block;
    // Periodic boundary condition is handled by the modulo operator
    int right_halo_idx = global_uint_idx % total_uints;
    s_data[uints_per_block + 1] = current_state[right_halo_idx];
  }

  __syncthreads(); // Synchronize to ensure all shared memory is loaded

  if (idx >= size)
    return; // Threads outside the working range terminate

  // --- 2. Get neighbor states from shared memory (with corrected boundary
  // logic) ---
  unsigned int left_val, center_val, right_val;

  // The center cell's value is always straightforward to find relative to the
  // thread's position in the block.
  int local_center_bit_idx = tid + 32;
  center_val = (s_data[local_center_bit_idx / 32] >>
                (31 - (local_center_bit_idx % 32))) &
               1;

  // Handle the left neighbor
  if (idx == 0) {
    // The left neighbor of cell 0 is cell (size-1).
    // Its bit position within its own uint is (size-1) % 32.
    // The uint containing cell (size-1) was loaded into the left halo,
    // s_data[0].
    int bit_pos = (size - 1) % 32;
    left_val = (s_data[0] >> (31 - bit_pos)) & 1;
  } else {
    // All other cells use the standard relative position.
    int local_left_bit_idx = tid + 31;
    left_val =
        (s_data[local_left_bit_idx / 32] >> (31 - (local_left_bit_idx % 32))) &
        1;
  }

  // Handle the right neighbor
  if (idx == size - 1) {
    // The right neighbor of cell (size-1) is cell 0.
    // Its bit position within its own uint is 0.
    // The uint containing cell 0 was loaded into the right halo for the block
    // containing the last cell.
    int right_halo_s_idx = uints_per_block + 1;
    right_val = (s_data[right_halo_s_idx] >> 31) & 1;
  } else {
    // All other cells use the standard relative position.
    int local_right_bit_idx = tid + 33;
    right_val = (s_data[local_right_bit_idx / 32] >>
                 (31 - (local_right_bit_idx % 32))) &
                1;
  }

  // Combine the 3 bits to get the neighborhood pattern (a number from 0 to 7)
  int neighborhood = (left_val << 2) | (center_val << 1) | right_val;

  // --- 3. Apply the rule and write the result atomically ---
  if (((rule >> neighborhood) & 1)) {
    // If the rule dictates this cell should be alive, set its bit to 1
    unsigned int mask = (1U << (31 - (idx % 32)));
    atomicOr(&next_state[idx / 32], mask);
  }
}

/**
 * @brief Block-level evolution kernel with internal iteration loop.
 *
 * This kernel implements a cellular automaton where each thread block operates
 * as an independent automaton. Key features for performance:
 * - Block-level boundary wrapping (no inter-block dependencies)
 * - All iterations executed in a single kernel launch
 * - Double buffering in shared memory (ping-pong)
 * - Minimized global memory access (only initial load and final write)
 * - No atomic operations needed (each thread writes to its own bit)
 *
 * @param state Global state buffer (input and final output)
 * @param temp_state Temporary global buffer (unused in this implementation)
 * @param rule Rule number (0-255)
 * @param num_steps Number of iterations to perform
 */
__global__ void evolve_block_level(unsigned int *state,
                                   unsigned int *temp_state, int rule,
                                   int num_steps) {
  extern __shared__ unsigned int s_mem[];

  int tid = threadIdx.x;
  int block_offset = blockIdx.x * blockDim.x;

  // Calculate number of uints needed for this block
  int uints_per_block = (blockDim.x + 31) / 32;

  // Set up double buffering pointers in shared memory
  unsigned int *s_current = s_mem;
  unsigned int *s_next = s_mem + uints_per_block;

  // --- 1. Load initial state from global to shared memory ---
  int uint_idx_in_block = tid / 32;
  int bit_idx_in_uint = tid % 32;

  // Cooperative loading: each warp loads one uint
  if (uint_idx_in_block < uints_per_block && tid < blockDim.x) {
    int global_uint_idx = (block_offset / 32) + uint_idx_in_block;
    if (bit_idx_in_uint == 0) {
      s_current[uint_idx_in_block] = state[global_uint_idx];
    }
  }

  __syncthreads();

  // --- 2. Perform iterations with block-level wrapping ---
  for (int iter = 0; iter < num_steps; iter++) {
    // Clear next buffer
    if (uint_idx_in_block < uints_per_block && bit_idx_in_uint == 0) {
      s_next[uint_idx_in_block] = 0;
    }
    __syncthreads();

    // Each thread processes one cell
    if (tid < blockDim.x) {
      unsigned int left_val, center_val, right_val;

      // Get center value
      int center_uint = tid / 32;
      int center_bit = tid % 32;
      center_val = (s_current[center_uint] >> (31 - center_bit)) & 1;

      // Get left neighbor with block-level wrapping
      int left_tid = (tid == 0) ? (blockDim.x - 1) : (tid - 1);
      int left_uint = left_tid / 32;
      int left_bit = left_tid % 32;
      left_val = (s_current[left_uint] >> (31 - left_bit)) & 1;

      // Get right neighbor with block-level wrapping
      int right_tid = (tid == blockDim.x - 1) ? 0 : (tid + 1);
      int right_uint = right_tid / 32;
      int right_bit = right_tid % 32;
      right_val = (s_current[right_uint] >> (31 - right_bit)) & 1;

      // Apply rule
      int neighborhood = (left_val << 2) | (center_val << 1) | right_val;

      // OPTIMIZATION: Use warp-sync bit manipulation to avoid atomicOr
      // contention
      unsigned int is_alive = ((rule >> neighborhood) & 1);
      unsigned int warp_mask = __ballot_sync(0xFFFFFFFF, is_alive);

      if (center_bit == 0) {
        s_next[center_uint] = warp_mask;
      }
    }

    __syncthreads();

    // Swap buffers (ping-pong)
    unsigned int *temp = s_current;
    s_current = s_next;
    s_next = temp;
  }

  // --- 3. Write final result back to global memory ---
  if (uint_idx_in_block < uints_per_block && bit_idx_in_uint == 0) {
    int global_uint_idx = (block_offset / 32) + uint_idx_in_block;
    state[global_uint_idx] = s_current[uint_idx_in_block];
  }
}
