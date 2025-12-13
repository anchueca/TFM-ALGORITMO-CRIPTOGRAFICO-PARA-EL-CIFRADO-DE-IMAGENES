/**
 * @file test_block_level.cu
 * @brief Test program to demonstrate and benchmark block-level cellular
 * automaton
 *
 * This program compares the performance and behavior of the standard iterate()
 * method versus the new iterate_block_level() method.
 */

#include "../include/automata.cuh"
#include <chrono>
#include <iomanip>
#include <iostream>

void print_comparison(const ElementalCelularAutomata &ca1,
                      const ElementalCelularAutomata &ca2,
                      size_t cells_to_show = 64) {
  std::vector<unsigned int> state1((cells_to_show + 31) / 32);
  std::vector<unsigned int> state2((cells_to_show + 31) / 32);

  size_t bytes_to_copy = state1.size() * sizeof(unsigned int);
  cudaMemcpy(state1.data(), ca1.get_cuda_state(), bytes_to_copy,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(state2.data(), ca2.get_cuda_state(), bytes_to_copy,
             cudaMemcpyDeviceToHost);

  std::cout << "Standard:    ";
  for (size_t i = 0; i < cells_to_show; i++) {
    unsigned int bit = (state1[i / 32] >> (31 - (i % 32))) & 1;
    std::cout << (bit ? '#' : '.');
  }
  std::cout << std::endl;

  std::cout << "Block-level: ";
  for (size_t i = 0; i < cells_to_show; i++) {
    unsigned int bit = (state2[i / 32] >> (31 - (i % 32))) & 1;
    std::cout << (bit ? '#' : '.');
  }
  std::cout << std::endl;
}

template <typename Func>
double benchmark(Func f, int warmup_runs = 3, int benchmark_runs = 10) {
  // Warmup
  for (int i = 0; i < warmup_runs; i++) {
    f();
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < benchmark_runs; i++) {
    f();
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration = end - start;
  return duration.count() / benchmark_runs;
}

int main() {
  std::cout << "========================================\n";
  std::cout << "  Block-Level Cellular Automaton Test\n";
  std::cout << "========================================\n\n";

  // Test parameters
  const size_t SIZE = 256 * 1024; // 256K cells
  const int RULE = 30;            // Rule 30 (chaotic)
  const int STEPS = 100;          // Number of iterations

  std::cout << "Configuration:\n";
  std::cout << "  Size: " << SIZE << " cells\n";
  std::cout << "  Rule: " << RULE << "\n";
  std::cout << "  Steps: " << STEPS << "\n";
  std::cout << "  Block Size: 256 threads\n\n";

  // Initialize CUDA context
  cudaFree(0);

  // Create initial state
  size_t num_uints = (SIZE + 31) / 32;
  std::vector<unsigned int> initial_state(num_uints, 0);
  // Set middle cell to 1
  int middle = SIZE / 2;
  initial_state[middle / 32] = 1U << (31 - (middle % 32));

  std::cout << "Test 1: Visual Comparison (first 64 cells after 10 steps)\n";
  std::cout << "-----------------------------------------------------------\n";

  // Create two automata with same initial state
  ElementalCelularAutomata ca_standard(initial_state, SIZE, RULE);
  ElementalCelularAutomata ca_block(initial_state, SIZE, RULE);

  ca_standard.iterate(10);
  ca_block.iterate_block_level(10);

  std::cout << "Note: Block-level shows independent evolution per block\n";
  std::cout << "      (256-cell blocks with wrap-around boundaries)\n\n";
  print_comparison(ca_standard, ca_block);

  std::cout << "\n";
  std::cout << "Test 2: Performance Benchmark\n";
  std::cout << "-----------------------------------------------------------\n";

  // Reset to initial state
  ElementalCelularAutomata ca_perf_standard(initial_state, SIZE, RULE);
  ElementalCelularAutomata ca_perf_block(initial_state, SIZE, RULE);

  // Benchmark standard method
  double time_standard = benchmark([&]() {
    ElementalCelularAutomata temp(initial_state, SIZE, RULE);
    temp.iterate(STEPS);
  });

  // Benchmark block-level method
  double time_block = benchmark([&]() {
    ElementalCelularAutomata temp(initial_state, SIZE, RULE);
    temp.iterate_block_level(STEPS);
  });

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Standard iterate():        " << time_standard << " ms\n";
  std::cout << "Block-level iterate():     " << time_block << " ms\n";
  std::cout << "Speedup:                   " << (time_standard / time_block)
            << "x\n";
  std::cout << "\nPerformance advantages of block-level approach:\n";
  std::cout << "  - Single kernel launch (vs " << STEPS << " launches)\n";
  std::cout << "  - No global memory writes between iterations\n";
  std::cout << "  - All computation in shared memory\n";
  std::cout << "  - Reduced kernel launch overhead\n";

  std::cout << "\n";
  std::cout << "Test 3: Scalability Test (varying iteration counts)\n";
  std::cout << "-----------------------------------------------------------\n";
  std::cout << std::setw(12) << "Iterations" << std::setw(15) << "Standard (ms)"
            << std::setw(15) << "Block (ms)" << std::setw(12) << "Speedup\n";
  std::cout << std::string(54, '-') << "\n";

  for (int steps : {10, 50, 100, 200, 500}) {
    double t_std = benchmark(
        [&]() {
          ElementalCelularAutomata temp(initial_state, SIZE, RULE);
          temp.iterate(steps);
        },
        2, 5);

    double t_blk = benchmark(
        [&]() {
          ElementalCelularAutomata temp(initial_state, SIZE, RULE);
          temp.iterate_block_level(steps);
        },
        2, 5);

    std::cout << std::setw(12) << steps << std::setw(15) << t_std
              << std::setw(15) << t_blk << std::setw(12) << (t_std / t_blk)
              << "x\n";
  }

  std::cout << "\n========================================\n";
  std::cout << "  All tests completed successfully!\n";
  std::cout << "========================================\n";

  return 0;
}
