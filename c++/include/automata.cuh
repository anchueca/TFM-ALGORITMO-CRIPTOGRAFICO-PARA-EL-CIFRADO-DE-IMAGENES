# ifndef AUTOMATA_CUH
# define AUTOMATA_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include "kernels_aux.cuh"

__global__ void evolve(unsigned int* current_state, unsigned int* next_state, size_t rule, size_t size, size_t num_steps);

__global__ void evolve_shared(const unsigned int* current_state, unsigned int* next_state, int rule, int size);

__global__ void sort_indices(unsigned int* indices, unsigned int* chaotic_values, size_t lenght);


class ElementalCelularAutomata {
private:
    int rule;
    size_t size;
    size_t size_in_bytes;

    int num_step;
    unsigned int *d_state[2] = {nullptr, nullptr};
    const size_t BLOCK_SIZE = 256;
    
public:
    ElementalCelularAutomata(size_t size, int rule);
    ElementalCelularAutomata(std::vector<unsigned int> initial_state, int size, int rule);
    ElementalCelularAutomata(std::vector<unsigned char> initial_state, int size, int rule);
    unsigned int* get_cuda_state() const;
    size_t get_size() const;
    size_t get_size_in_bytes() const;


    ~ElementalCelularAutomata();

    void iterate(int num_steps = 1);
    void print_state() const;
    void show();
};

# endif // AUTOMATA_CUH