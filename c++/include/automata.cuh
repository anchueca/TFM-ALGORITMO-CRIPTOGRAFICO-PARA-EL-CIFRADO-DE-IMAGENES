#ifndef AUTOMATA_CUH
#define AUTOMATA_CUH

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>
#include <limits>

/**
 * @class ElementalCelularAutomata
 * @brief Implements a 1D elementary cellular automaton using CUDA for high performance.
 *
 * The automaton's state is a bit array, packed into an array of unsigned ints
 * to optimize memory usage and transfers. It uses a double-buffering (ping-pong) 
 * strategy for iterations and shared memory in the kernel to speed up access 
 * to each cell's neighbors.
 */
class ElementalCelularAutomata {
public:
    // --- Constructors ---

    /**
     * @brief Constructor that initializes the automaton with a random state.
     * @param size The total number of cells (bits) in the automaton.
     * @param rule The rule number (0-255) to apply.
     */
    ElementalCelularAutomata(size_t size, int rule);

    /**
     * @brief Constructor that initializes the automaton from a predefined, pre-packed state (vector of unsigned int).
     * @param initial_state Vector with the initial state already bit-packed into integers.
     * @param size The total number of cells represented by the packed data.
     * @param rule The rule number.
     */
    ElementalCelularAutomata(const std::vector<unsigned int>& initial_state, size_t size, int rule);
    
    /**
     * @brief Constructor that initializes the automaton from a predefined, pre-packed state (vector of unsigned char).
     * @param initial_state Vector with the initial state already bit-packed into bytes.
     * @param size The total number of cells represented by the packed data.
     * @param rule The rule number.
     */
    ElementalCelularAutomata(const std::vector<unsigned char>& initial_state, size_t size, int rule);

    /**
     * @brief Constructor that initializes the automaton from a predefined, pre-packed state in vram.
     * @param initial_state Pointer to the initial state already in vram.
     * @param size The total number of cells represented by the packed data.
     * @param rule The rule number.
     */
    ElementalCelularAutomata(unsigned int* cuda_pointer, size_t size, int rule);


    /**
     * @brief Destructor. Frees the GPU memory.
     */
    ~ElementalCelularAutomata();

    // --- Public Methods ---

    /**
     * @brief Evolves the automaton's state for a given number of steps.
     * @param num_steps The number of iterations to perform.
     */
    void iterate(int num_steps = 1);

    /**
     * @brief Prints the current state of the automaton to the console.
     * '#' represents a live cell, a blank space ' ' a dead cell.
     */
    void print_state() const;

    /**
     * @brief Iterates and prints the state at each step, a specified number of times.
     * @param times The number of generations to display.
     */
    void print_states(size_t times);
    
    /**
     * @brief Prints the `unsigned int` integer values that make up the state for debugging purposes.
     */
    void print_state_int() const;

    // --- Getters ---
    const unsigned int* get_cuda_state() const;
    size_t get_size() const;
    size_t get_size_in_bytes() const;

private:
    // --- Private Members ---
    size_t size;            // Total number of cells (bits)
    size_t size_in_bytes;   // Total size in bytes of the state in memory
    int rule;               // Automaton rule (0-255)

    // Pointers to the state in GPU memory (double buffer)
    unsigned int* d_state[2];

    // Constant for the thread block size
    static const int BLOCK_SIZE = 256;
};

// --- CUDA Kernel Declaration ---
__global__ void evolve_shared(const unsigned int* current_state, unsigned int* next_state, int rule, int size);

#endif // AUTOMATA_CUH

