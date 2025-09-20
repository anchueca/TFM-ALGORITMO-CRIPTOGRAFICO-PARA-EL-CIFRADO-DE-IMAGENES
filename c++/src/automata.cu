/*#include "../include/automata.cuh"

    ElementalCelularAutomata::ElementalCelularAutomata(int size, int rule) : size(size), rule(rule), num_step(0), current_index(0) {
        state = new unsigned char[size];
        cudaMalloc(&d_state[0], size * sizeof(unsigned char));
        cudaMalloc(&d_state[1], size * sizeof(unsigned char));

        // Inicializar el estado de manera aleatoria (PRUEBAS)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, 1);

        for (int i = 0; i < size; i++) {
            state[i] = dist(gen);
        }

        // Copiar el estado inicial a la memoria de la GPU
        cudaMemcpy(d_state[0], state, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    }

    ElementalCelularAutomata::~ElementalCelularAutomata() {
        delete[] state;
        cudaFree(d_state[0]);
        cudaFree(d_state[1]);
    }

    void ElementalCelularAutomata::step_cuda(int num_steps = 1) {
        for (int step = 0; step < num_steps; step++) {
            int next_index = (current_index + 1) % 2;

            // Llamada al kernel de CUDA
            evolve<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_state[current_index], d_state[next_index], rule, size);
            cudaDeviceSynchronize();

            // Copiar el estado de vuelta a la memoria del host
            cudaMemcpy(state, d_state[next_index], size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            current_index = next_index;
            num_step++;
        }
    }

    std::vector<unsigned char> ElementalCelularAutomata::convert_to_bitstream() {
        std::vector<unsigned char> bitstream((size + 7) / 8, 0);

        for (int i = 0; i < size; i++) {
            bitstream[i / 8] |= (state[i] << (7 - i % 8));
        }

        return bitstream;
    }*/