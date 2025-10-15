#include "../include/automata.cuh"

ElementalCelularAutomata::ElementalCelularAutomata(size_t size, int rule) : size(size), rule(rule) {
    this->size_in_bytes = (this->size + 31) / 32 * sizeof(unsigned int); 

    std::vector<unsigned int> h_state((this->size + 31) / 32);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dist(0, std::numeric_limits<unsigned int>::max());

    for (int i = 0; i < (this->size + 31) / 32; i++) {
        h_state[i] = dist(gen);
    }

    cudaMalloc(&this->d_state[0], this->size_in_bytes);
    cudaMalloc(&this->d_state[1], this->size_in_bytes);
    cudaMemcpy(this->d_state[0], h_state.data(), this->size_in_bytes, cudaMemcpyHostToDevice);
}

ElementalCelularAutomata::ElementalCelularAutomata(std::vector<unsigned int> initial_state, int size, int rule)
    : size(size), rule(rule) {
    this->size_in_bytes = (this->size + 31) / 32 * sizeof(unsigned int); 
    cudaMalloc(&this->d_state[0], this->size_in_bytes);
    cudaMalloc(&this->d_state[1], this->size_in_bytes);

    cudaMemcpy(this->d_state[0], initial_state.data(), size_in_bytes, cudaMemcpyHostToDevice);
}

ElementalCelularAutomata::ElementalCelularAutomata(std::vector<unsigned char> initial_state, int size, int rule)
    : size(size), rule(rule) {
    this->size_in_bytes = (this->size + 7) / 8 * sizeof(unsigned int); 
    cudaMalloc(&this->d_state[0], this->size_in_bytes);
    cudaMalloc(&this->d_state[1], this->size_in_bytes);

    cudaMemcpy(this->d_state[0], initial_state.data(), this->size_in_bytes, cudaMemcpyHostToDevice);
}

ElementalCelularAutomata::~ElementalCelularAutomata() {
    cudaFree(this->d_state[0]);
    cudaFree(this->d_state[1]);
}

void ElementalCelularAutomata::iterate(int num_steps) {
    int num_blocks = (this->size + this->BLOCK_SIZE - 1) / this->BLOCK_SIZE;

    // Shared memory size: (bytes per block) + 2 halo bytes
    size_t shared_mem_size = (this->BLOCK_SIZE / 32 + 2) * sizeof(unsigned int);

    for (int i = 0; i < num_steps; ++i) {
        // Clear the destination buffer before the kernel writes to it
        cudaMemset(this->d_state[1], 0, this->size_in_bytes);

        // Launch kernel with dynamic shared memory
        evolve_shared<<<num_blocks, this->BLOCK_SIZE, shared_mem_size>>>(
            this->d_state[0], this->d_state[1], this->rule, this->size);
        
        // Ping-pong buffering: swap pointers for the next iteration
        unsigned int* temp = this->d_state[0];
        this->d_state[0] = this->d_state[1];
        this->d_state[1] = temp;
    }
    cudaDeviceSynchronize();
}

void ElementalCelularAutomata::print_state() const{
    size_t num_uints = (this->size + 31) / 32;

    std::vector<unsigned int> h_state(num_uints);
    cudaMemcpy(h_state.data(), this->d_state[0], this->size_in_bytes, cudaMemcpyDeviceToHost);

    int bits_printed = 0;
    for (size_t i = 0; i < num_uints; ++i) {
        for (int bit = 0; bit < 32; ++bit) {
            // guard to prevent printing more bits than 'size'
            if (bits_printed >= this->size) break;
            
            unsigned char bit_value = (h_state[i] >> (31 - bit)) & 1;
            std::cout << (bit_value ? '1' : ' ');
            bits_printed++;
        }
    }
    std::cout << std::endl;
}

// High-performance kernel using shared memory
__global__ void evolve_shared(const unsigned int* current_state, unsigned int* next_state, int rule, int size) {
    extern __shared__ unsigned int s_data[];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int idx = block_offset + tid;

    // --- 1. Load data from global to shared memory (Corrected Halo Logic) ---
    int uints_per_block = blockDim.x / 32;
    int shared_mem_idx = tid / 32;
    int total_uints = (size + 31) / 32;

    if (shared_mem_idx < uints_per_block) {
        int global_read_idx = (block_offset / 32) + shared_mem_idx;
        if(global_read_idx < total_uints)
            s_data[shared_mem_idx + 1] = current_state[global_read_idx];
    }
    if (tid == 0) {
        int global_uint_idx = block_offset / 32;
        int left_halo_idx = (global_uint_idx == 0) ? total_uints - 1 : global_uint_idx - 1;
        s_data[0] = current_state[left_halo_idx];
    }
    if (tid == blockDim.x - 1) {
        int global_uint_idx = (block_offset / 32) + uints_per_block;
        int right_halo_idx = (global_uint_idx >= total_uints) ? 0 : global_uint_idx;
        if(right_halo_idx < total_uints)
            s_data[uints_per_block + 1] = current_state[right_halo_idx];
    }

    __syncthreads(); // Synchronize block to ensure all data is loaded

    if (idx >= size) return;

    // --- 2. Get neighbor states from FAST shared memory ---
    // It maps the local thread index
    // to the bit's position within the shared memory tile (including halos).
    // The bit at local index 'i' is in s_data[i/32] at bit 'i%32'.
    // We add 32 to our local index 'tid' to account for the left halo (32 bits).
    int local_left_bit_idx = tid + 31;
    int local_center_bit_idx = tid + 32;
    int local_right_bit_idx = tid + 33;
    
    unsigned int left_val   = (s_data[local_left_bit_idx / 32] >> (31 - (local_left_bit_idx % 32))) & 1;
    unsigned int center_val = (s_data[local_center_bit_idx / 32] >> (31 - (local_center_bit_idx % 32))) & 1;
    unsigned int right_val  = (s_data[local_right_bit_idx / 32] >> (31 - (local_right_bit_idx % 32))) & 1;

    int neighborhood = (left_val << 2) | (center_val << 1) | right_val;

    // --- 3. Apply rule and write result atomically ---
    if (((rule >> neighborhood) & 1)) {
        unsigned int mask = (1U << (31 - (idx % 32)));
        atomicOr(&next_state[idx / 32], mask);
    }
}

void ElementalCelularAutomata::show() {
    for(int i=0; i<50; i++) {
        this->iterate(1);
        this->print_state();
    }
}

unsigned int* ElementalCelularAutomata::get_cuda_state() const{
    return this->d_state[0];
}

size_t ElementalCelularAutomata::get_size() const{
    return this->size;
}

size_t ElementalCelularAutomata::get_size_in_bytes() const{
    return this->size_in_bytes;
}