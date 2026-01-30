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
  cudaError_t err;
  err = cudaMalloc(&this->d_state[0], this->size_in_bytes);
  if (err != cudaSuccess) throw std::runtime_error("ECA cudaMalloc 0 failed: " + std::string(cudaGetErrorString(err)));
  err = cudaMalloc(&this->d_state[1], this->size_in_bytes);
  if (err != cudaSuccess) throw std::runtime_error("ECA cudaMalloc 1 failed: " + std::string(cudaGetErrorString(err)));
  
  err = cudaMemcpy(this->d_state[0], h_state.data(), this->size_in_bytes,
             cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw std::runtime_error("ECA cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
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

  cudaError_t err;
  err = cudaMalloc(&this->d_state[0], this->size_in_bytes);
  if (err != cudaSuccess) throw std::runtime_error("ECA cudaMalloc 0 failed: " + std::string(cudaGetErrorString(err)));
  err = cudaMalloc(&this->d_state[1], this->size_in_bytes);
  if (err != cudaSuccess) throw std::runtime_error("ECA cudaMalloc 1 failed: " + std::string(cudaGetErrorString(err)));

  err = cudaMemcpy(this->d_state[0], initial_state.data(), size_in_bytes,
             cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw std::runtime_error("ECA cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw std::runtime_error("ECA cudaDeviceSynchronize failed: " + std::string(cudaGetErrorString(err)));
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

  cudaError_t err;
  err = cudaMalloc(&this->d_state[0], this->size_in_bytes);
  if (err != cudaSuccess) throw std::runtime_error("ECA cudaMalloc 0 failed: " + std::string(cudaGetErrorString(err)));
  err = cudaMalloc(&this->d_state[1], this->size_in_bytes);
  if (err != cudaSuccess) throw std::runtime_error("ECA cudaMalloc 1 failed: " + std::string(cudaGetErrorString(err)));

  err = cudaMemcpy(this->d_state[0], initial_state.data(), this->size_in_bytes,
             cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw std::runtime_error("ECA cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
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
unsigned int *ElementalCelularAutomata::get_cuda_state() const {
  return this->d_state[0];
}

size_t ElementalCelularAutomata::get_size() const { return this->size; }

// Return size in bytes of the packed state (implementation)
size_t ElementalCelularAutomata::get_size_in_bytes() const {
  return this->size_in_bytes;
}
