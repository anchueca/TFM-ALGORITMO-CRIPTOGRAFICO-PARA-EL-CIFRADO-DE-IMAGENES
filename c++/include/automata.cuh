# ifndef AUTOMATA_CUH
# define AUTOMATA_CUH
/*
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

class ElementalCelularAutomata {
private:
    int rule;
    int size;
    int num_step;
    unsigned char *state;
    unsigned char *d_state[2];
    int current_index;
    
public:
    ElementalCelularAutomata(int size, int rule) : size(size), rule(rule), num_step(0), current_index(0);

    ~ElementalCelularAutomata();

    void step_cuda(int num_steps = 1);
    std::vector<unsigned char> convert_to_bitstream();
};
*/
# endif // AUTOMATA_CUH