#include "../include/automata.cuh"

// --- Constructors ---

ElementalCelularAutomata::ElementalCelularAutomata(size_t size, int rule)
    : size(size),
      rule(rule),
      size_in_bytes((size + 31) / 32 * sizeof(unsigned int)) {

    // Create a random state on the host
    size_t num_uints = size_in_bytes / sizeof(unsigned int);
    std::vector<unsigned int> h_state(num_uints);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dist(0, std::numeric_limits<unsigned int>::max());

    for (size_t i = 0; i < num_uints; i++) {
        h_state[i] = dist(gen);
    }
    
    // Force the last integer to have zeros in the unused bits to avoid wrap-around issues
    int remaining_bits = size % 32;
    if (remaining_bits > 0) {
        // Create a mask to clear the unused bits at the end of the last integer
        unsigned int mask = (1U << (32 - remaining_bits)) - 1;
        h_state.back() &= ~mask;
    }

    // Allocate memory on the GPU and copy the initial state
    cudaMalloc(&this->d_state[0], this->size_in_bytes);
    cudaMalloc(&this->d_state[1], this->size_in_bytes);
    cudaMemcpy(this->d_state[0], h_state.data(), this->size_in_bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

ElementalCelularAutomata::ElementalCelularAutomata(const std::vector<unsigned int>& initial_state, size_t size, int rule)
    : size(size),
      rule(rule),
      size_in_bytes((size + 31) / 32 * sizeof(unsigned int)) {

    // Throw an exception with a detailed message if the size does not match.
    size_t received_size_in_bytes = initial_state.size() * sizeof(unsigned int);
    if (received_size_in_bytes != this->size_in_bytes) {
        std::string error_message = "Error: Mismatch in initialization vector size. Expected " +
                                    std::to_string(this->size_in_bytes) + " bytes, but received " +
                                    std::to_string(received_size_in_bytes) + " bytes.";
        throw std::invalid_argument(error_message);
    }

    cudaMalloc(&this->d_state[0], this->size_in_bytes);
    cudaMalloc(&this->d_state[1], this->size_in_bytes);

    cudaMemcpy(this->d_state[0], initial_state.data(), size_in_bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

// Constructor that initializes from a pre-packed vector of bytes.
ElementalCelularAutomata::ElementalCelularAutomata(const std::vector<unsigned char>& initial_state, size_t size, int rule)
    : size(size),
      rule(rule),
      size_in_bytes((size + 31) / 32 * sizeof(unsigned int)) {
    
    // Throw an exception with a detailed message if the size does not match.
    if (initial_state.size() != this->size_in_bytes) {
        std::string error_message = "Error: Mismatch in initialization vector size. Expected " +
                                    std::to_string(this->size_in_bytes) + " bytes, but received " +
                                    std::to_string(initial_state.size()) + " bytes.";
        throw std::invalid_argument(error_message);
    }

    cudaMalloc(&this->d_state[0], this->size_in_bytes);
    cudaMalloc(&this->d_state[1], this->size_in_bytes);

    cudaMemcpy(this->d_state[0], initial_state.data(), this->size_in_bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

ElementalCelularAutomata::ElementalCelularAutomata(unsigned int* cuda_pointer, size_t size, int rule)
    : size(size),
      rule(rule),
      size_in_bytes((size + 31) / 32 * sizeof(unsigned int)) {

    this->d_state[0] = cuda_pointer;
    cudaMalloc(&this->d_state[1], this->size_in_bytes);
}

// --- Destructor ---

ElementalCelularAutomata::~ElementalCelularAutomata() {
    // Check for nullptrs in case a constructor failed after the size check
    if(d_state[0]) cudaFree(this->d_state[0]);
    if(d_state[1]) cudaFree(this->d_state[1]);
}

// --- Methods ---

void ElementalCelularAutomata::iterate(int num_steps) {
    if (size == 0) return;

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
        unsigned int* temp = this->d_state[0];
        this->d_state[0] = this->d_state[1];
        this->d_state[1] = temp;
    }
    cudaDeviceSynchronize();
}

void ElementalCelularAutomata::print_state() const {
    if (size == 0) return;

    size_t num_uints = (this->size + 31) / 32;

    std::vector<unsigned int> h_state(num_uints);
    cudaMemcpy(h_state.data(), this->d_state[0], this->size_in_bytes, cudaMemcpyDeviceToHost);

    int bits_printed = 0;
    for (size_t i = 0; i < num_uints; ++i) {
        for (int bit = 0; bit < 32; ++bit) {
            if (bits_printed >= this->size) break;

            unsigned char bit_value = (h_state[i] >> (31 - bit)) & 1;
            std::cout << (bit_value ? '#' : ' '); // Use '#' for better visibility
            bits_printed++;
        }
    }
    std::cout << std::endl;
}

void ElementalCelularAutomata::print_states(size_t times) {
    for (size_t i = 0; i < times; i++) {
        this->print_state();
        this->iterate(1);
    }
}

void ElementalCelularAutomata::print_state_int() const {
    if (size == 0) return;
    
    size_t num_uints = (this->size + 31) / 32;

    std::vector<unsigned int> h_state(num_uints);
    cudaMemcpy(h_state.data(), this->d_state[0], this->size_in_bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < num_uints; i++) {
        printf("idx %zu: %u\n", i, h_state[i]);
    }
}

// --- Getters ---

const unsigned int* ElementalCelularAutomata::get_cuda_state() const {
    return this->d_state[0];
}

size_t ElementalCelularAutomata::get_size() const {
    return this->size;
}

size_t ElementalCelularAutomata::get_size_in_bytes() const {
    return this->size_in_bytes;
}

// --- CUDA Kernel ---

__global__ void evolve_shared(const unsigned int* current_state, unsigned int* next_state, int rule, int size) {
    // Shared memory is declared externally; its size is passed at launch
    extern __shared__ unsigned int s_data[];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int idx = block_offset + tid;

    // --- 1. Load data from global to shared memory (with halos) ---
    int uints_per_block = blockDim.x / 32;
    int shared_mem_idx = tid / 32;
    int total_uints = (size + 31) / 32;

    // Each warp loads one unsigned int from the global state to the center of shared memory
    if (shared_mem_idx < uints_per_block) {
        int global_read_idx = (block_offset / 32) + shared_mem_idx;
        if (global_read_idx < total_uints) {
            s_data[shared_mem_idx + 1] = current_state[global_read_idx];
        }
    }

    // The first thread of the block loads the left halo (with periodic boundary condition)
    if (tid == 0) {
        int global_uint_idx = block_offset / 32;
        int left_halo_idx = (global_uint_idx == 0) ? total_uints - 1 : global_uint_idx - 1;
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

    if (idx >= size) return; // Threads outside the working range terminate

    // --- 2. Get neighbor states from shared memory (with corrected boundary logic) ---
    unsigned int left_val, center_val, right_val;

    // The center cell's value is always straightforward to find relative to the thread's position in the block.
    int local_center_bit_idx = tid + 32;
    center_val = (s_data[local_center_bit_idx / 32] >> (31 - (local_center_bit_idx % 32))) & 1;

    // Handle the left neighbor
    if (idx == 0) {
        // The left neighbor of cell 0 is cell (size-1).
        // Its bit position *within its own uint* is (size-1) % 32.
        // The uint containing cell (size-1) was loaded into the left halo, s_data[0].
        int bit_pos = (size - 1) % 32;
        left_val = (s_data[0] >> (31 - bit_pos)) & 1;
    } else {
        // All other cells use the standard relative position.
        int local_left_bit_idx = tid + 31;
        left_val = (s_data[local_left_bit_idx / 32] >> (31 - (local_left_bit_idx % 32))) & 1;
    }

    // Handle the right neighbor
    if (idx == size - 1) {
        // The right neighbor of cell (size-1) is cell 0.
        // Its bit position *within its own uint* is 0.
        // The uint containing cell 0 was loaded into the right halo for the block containing the last cell.
        int right_halo_s_idx = uints_per_block + 1;
        right_val = (s_data[right_halo_s_idx] >> 31) & 1;
    } else {
        // All other cells use the standard relative position.
        int local_right_bit_idx = tid + 33;
        right_val = (s_data[local_right_bit_idx / 32] >> (31 - (local_right_bit_idx % 32))) & 1;
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