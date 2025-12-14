/**
 * @file automata.cu
 * @brief Implementation of the ElementalCelularAutomata class and its CUDA
 * kernel.
 *
 * This file provides constructors that initialize the automata state on the
 * GPU, methods to iterate the automata, debug print helpers and the CUDA
 * kernel that performs one evolution step using shared memory for neighbor
 * access.
 */

#include "../include/automata.cuh"

// --- Constructors (brief implementation notes) ---
ElementalCelularAutomata::ElementalCelularAutomata(size_t size, int rule)
    : size(size), rule(rule),
      size_in_bytes((size + 31) / 32 * sizeof(unsigned int)) {

  // Create a random state on the host
  size_t num_uints = size_in_bytes / sizeof(unsigned int);
  std::vector<unsigned int> h_state(num_uints);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> dist(
      0, std::numeric_limits<unsigned int>::max());

  for (size_t i = 0; i < num_uints; i++) {
    h_state[i] = dist(gen);
  }

  // Force the last integer to have zeros in the unused bits to avoid
  // wrap-around issues
  int remaining_bits = size % 32;
  if (remaining_bits > 0) {
    // Create a mask to clear the unused bits at the end of the last integer
    unsigned int mask = (1U << (32 - remaining_bits)) - 1;
    h_state.back() &= ~mask;
  }

  // Allocate memory on the GPU and copy the initial state
  cudaMalloc(&this->d_state[0], this->size_in_bytes);
  cudaMalloc(&this->d_state[1], this->size_in_bytes);
  cudaMemcpy(this->d_state[0], h_state.data(), this->size_in_bytes,
             cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

// Construct from a pre-packed vector (see header for API details)
ElementalCelularAutomata::ElementalCelularAutomata(
    const std::vector<unsigned int> &initial_state, size_t size, int rule)
    : size(size), rule(rule),
      size_in_bytes((size + 31) / 32 * sizeof(unsigned int)) {

  // Throw an exception with a detailed message if the size does not match.
  size_t received_size_in_bytes = initial_state.size() * sizeof(unsigned int);
  if (received_size_in_bytes != this->size_in_bytes) {
    std::string error_message =
        "Error: Mismatch in initialization vector size. Expected " +
        std::to_string(this->size_in_bytes) + " bytes, but received " +
        std::to_string(received_size_in_bytes) + " bytes.";
    throw std::invalid_argument(error_message);
  }

  cudaMalloc(&this->d_state[0], this->size_in_bytes);
  cudaMalloc(&this->d_state[1], this->size_in_bytes);

  cudaMemcpy(this->d_state[0], initial_state.data(), size_in_bytes,
             cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

// Construct from a pre-packed byte vector (see header for API details)
ElementalCelularAutomata::ElementalCelularAutomata(
    const std::vector<unsigned char> &initial_state, size_t size, int rule)
    : size(size), rule(rule),
      size_in_bytes((size + 31) / 32 * sizeof(unsigned int)) {

  // Throw an exception with a detailed message if the size does not match.
  if (initial_state.size() != this->size_in_bytes) {
    std::string error_message =
        "Error: Mismatch in initialization vector size. Expected " +
        std::to_string(this->size_in_bytes) + " bytes, but received " +
        std::to_string(initial_state.size()) + " bytes.";
    throw std::invalid_argument(error_message);
  }

  cudaMalloc(&this->d_state[0], this->size_in_bytes);
  cudaMalloc(&this->d_state[1], this->size_in_bytes);

  cudaMemcpy(this->d_state[0], initial_state.data(), this->size_in_bytes,
             cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

// Construct using an existing device pointer (see header for API details)
ElementalCelularAutomata::ElementalCelularAutomata(unsigned int *cuda_pointer,
                                                   size_t size, int rule)
    : size(size), rule(rule),
      size_in_bytes((size + 31) / 32 * sizeof(unsigned int)) {

  this->d_state[0] = cuda_pointer;
  cudaMalloc(&this->d_state[1], this->size_in_bytes);
}

// --- Destructor (implementation) ---
// See header for documentation of lifecycle behavior.
ElementalCelularAutomata::~ElementalCelularAutomata() {
  // Check for nullptrs in case a constructor failed after the size check
  if (d_state[0])
    cudaFree(this->d_state[0]);
  if (d_state[1])
    cudaFree(this->d_state[1]);
}

// --- Methods (implementations) ---
// The detailed API docs for these methods live in the header file.
void ElementalCelularAutomata::iterate(int num_steps) {
  if (size == 0)
    return;

  int num_blocks = (this->size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Shared memory size: (uints per block) + 2 uints for halos
  size_t shared_mem_size = (BLOCK_SIZE / 32 + 2) * sizeof(unsigned int);

  for (int i = 0; i < num_steps; ++i) {
    // Clear the destination buffer before the kernel writes to it
    cudaMemset(this->d_state[1], 0, this->size_in_bytes);
    // Launch the kernel with dynamic shared memory
    evolve_shared<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        this->d_state[0], this->d_state[1], this->rule, this->size);

    // Swap pointers (ping-pong) for the next iteration
    unsigned int *temp = this->d_state[0];
    this->d_state[0] = this->d_state[1];
    this->d_state[1] = temp;
  }
  cudaDeviceSynchronize();
}

// Block-level iteration implementation
void ElementalCelularAutomata::iterate_block_level(int num_steps) {
  if (size == 0)
    return;

  // Dynamic thread configuration based on automata size
  int threads_per_block;
  if (size <= 512) {
    threads_per_block = 256; // Small: standard config
  } else if (size <= 2048) {
    threads_per_block = 512; // Medium: better occupancy
  } else {
    threads_per_block = 1024; // Large: maximum utilization
  }

  int num_blocks = (this->size + threads_per_block - 1) / threads_per_block;

  // Shared memory: double buffering within shared memory for ping-pong
  // We need 2 buffers, each holding (uints per block) unsigned ints
  int uints_per_block = (threads_per_block + 31) / 32;
  size_t shared_mem_size = 2 * uints_per_block * sizeof(unsigned int);

  // Launch single kernel that performs all iterations internally
  evolve_block_level<<<num_blocks, threads_per_block, shared_mem_size>>>(
      this->d_state[0], this->d_state[1], this->rule, num_steps);

  cudaDeviceSynchronize();
}

// Print the automaton state (implementation; see header for docs)
void ElementalCelularAutomata::print_state() const {
  if (size == 0)
    return;

  size_t num_uints = (this->size + 31) / 32;

  std::vector<unsigned int> h_state(num_uints);
  cudaMemcpy(h_state.data(), this->d_state[0], this->size_in_bytes,
             cudaMemcpyDeviceToHost);

  int bits_printed = 0;
  for (size_t i = 0; i < num_uints; ++i) {
    for (int bit = 0; bit < 32; ++bit) {
      if (bits_printed >= this->size)
        break;

      unsigned char bit_value = (h_state[i] >> (31 - bit)) & 1;
      std::cout << (bit_value ? '#' : ' '); // Use '#' for better visibility
      bits_printed++;
    }
  }
  std::cout << std::endl;
}

// Iterate and print states (implementation)
void ElementalCelularAutomata::print_states(size_t times) {
  for (size_t i = 0; i < times; i++) {
    this->print_state();
    this->iterate(1);
  }
}

// Print internal packed integer state (implementation)
void ElementalCelularAutomata::print_state_int() const {
  if (size == 0)
    return;

  size_t num_uints = (this->size + 31) / 32;

  std::vector<unsigned int> h_state(num_uints);
  cudaMemcpy(h_state.data(), this->d_state[0], this->size_in_bytes,
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < num_uints; i++) {
    printf("idx %zu: %u\n", i, h_state[i]);
  }
}

// --- Getters (implementations) ---
const unsigned int *ElementalCelularAutomata::get_cuda_state() const {
  return this->d_state[0];
}

size_t ElementalCelularAutomata::get_size() const { return this->size; }

// Return size in bytes of the packed state (implementation)
size_t ElementalCelularAutomata::get_size_in_bytes() const {
  return this->size_in_bytes;
}

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
      if ((rule >> neighborhood) & 1) {
        // Set bit in next state
        // MUST use atomic: concurrent writes from multiple threads
        // ~2% overhead but ensures correctness
        unsigned int mask = (1U << (31 - center_bit));
        atomicOr(&s_next[center_uint], mask);
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