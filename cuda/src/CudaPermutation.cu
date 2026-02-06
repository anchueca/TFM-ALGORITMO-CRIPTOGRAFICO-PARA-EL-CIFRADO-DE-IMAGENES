#include "../include/CudaPermutation.cuh"

// Macro to check CUDA errors
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

// --- HELPER FUNCTIONS ---

int next_power_of_2(int n) {
  if (n == 0)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

// --- SINGLE ARRAY SORT KERNELS ---

__global__ void init_buffers_kernel(float *values, unsigned int *indices, int n,
                                    int padded_size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < padded_size) {
    // Initialize indices: identity
    indices[idx] = idx;

    // Padding handling for values
    if (idx >= n) {
      values[idx] =
          FLT_MAX; // Infinity, so they end up at the end after sorting
    }
    // Note: Values < n were already copied via cudaMemcpy
  }
}

__global__ void bitonic_sort_step_kernel(float *values, unsigned int *indices,
                                         int j, int k, int padded_size) {
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int ixj = i ^ j;

  // Process if ixj > i to avoid duplication and stay within range
  if (ixj > i && ixj < padded_size) {
    float v1 = values[i];
    float v2 = values[ixj];

    // Sort direction:
    // (i & k) == 0 -> Ascending
    // (i & k) != 0 -> Descending
    bool ascending = ((i & k) == 0);

    // Swap logic
    if ((ascending && v1 > v2) || (!ascending && v1 < v2)) {
      // Swap values (keys)
      values[i] = v2;
      values[ixj] = v1;

      // Swap indices (payload)
      unsigned int temp_idx = indices[i];
      indices[i] = indices[ixj];
      indices[ixj] = temp_idx;
    }
  }
}

// --- SINGLE ARRAY IMPLEMENTATION ---

void compute_permutation_gpu(const float *h_chaotic_sequence,
                             int *h_permutation, int n) {
  int padded_size = next_power_of_2(n);

  // Device pointers
  float *d_values;
  unsigned int *d_indices;

  // 1. GPU Memory Allocation (Padded size)
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_values, padded_size * sizeof(float)));
  CHECK_CUDA_ERROR(
      cudaMalloc((void **)&d_indices, padded_size * sizeof(unsigned int)));

  // 2. Copy input data (only the n real data points)
  CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_chaotic_sequence, n * sizeof(float),
                              cudaMemcpyHostToDevice));

  // 3. Execution Configuration
  int threadsPerBlock = 256;
  int blocksPerGrid = (padded_size + threadsPerBlock - 1) / threadsPerBlock;

  // 4. Initialize padding and indices on GPU
  init_buffers_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_values, d_indices,
                                                          n, padded_size);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 5. Bitonic Sort Loop
  // k determines the size of the monotonic subsequence
  // j determines the comparison stride
  for (int k = 2; k <= padded_size; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonic_sort_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          d_values, d_indices, j, k, padded_size);
    }
  }

  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 6. Copy results back (only the first n indices)
  // Since we used FLT_MAX for padding and sorted ascendingly,
  // the valid indices are at the beginning. The padding ended up at [n ...
  // padded_size-1].
  CHECK_CUDA_ERROR(cudaMemcpy(h_permutation, d_indices,
                              n * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

  // 7. Free memory
  CHECK_CUDA_ERROR(cudaFree(d_values));
  CHECK_CUDA_ERROR(cudaFree(d_indices));
}

// --- NEW DEVICE-ONLY IMPLEMENTATION ---

void compute_permutation_device(float *d_values, unsigned int *d_indices,
                                int n) {
  int padded_size = next_power_of_2(n);

  float *d_padded_values = nullptr;
  unsigned int *d_padded_indices = nullptr;

  // Allocate temporary padded buffers
  CHECK_CUDA_ERROR(
      cudaMalloc((void **)&d_padded_values, padded_size * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_padded_indices,
                              padded_size * sizeof(unsigned int)));

  // Copy original values to padded buffer
  CHECK_CUDA_ERROR(cudaMemcpy(d_padded_values, d_values, n * sizeof(float),
                              cudaMemcpyDeviceToDevice));

  int threadsPerBlock = 256;
  int blocksPerGrid = (padded_size + threadsPerBlock - 1) / threadsPerBlock;

  // Initialize padding and indices
  init_buffers_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_padded_values, d_padded_indices, n, padded_size);

  // Parallel Bitonic Sort
  for (int k = 2; k <= padded_size; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonic_sort_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          d_padded_values, d_padded_indices, j, k, padded_size);
    }
  }

  // Copy results back (only the first n elements)
  CHECK_CUDA_ERROR(cudaMemcpy(d_indices, d_padded_indices,
                              n * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));

  // Sync to ensure results are ready
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(cudaFree(d_padded_values));
  CHECK_CUDA_ERROR(cudaFree(d_padded_indices));
}

// --- BATCHED SORT KERNELS ---

__global__ void copy_to_padded_kernel(const unsigned short *input_vals,
                                      const unsigned int *input_idxs,
                                      int *padded_vals,
                                      unsigned int *padded_idxs, int valid_len,
                                      int padded_len, int total_blocks) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < total_blocks * padded_len) {
    int block_id = tid / padded_len;
    int offset_in_block = tid % padded_len;

    int global_padded_idx = block_id * padded_len + offset_in_block;

    if (offset_in_block < valid_len) {
      int global_input_idx = block_id * valid_len + offset_in_block;
      padded_vals[global_padded_idx] = (int)input_vals[global_input_idx];
      padded_idxs[global_padded_idx] = input_idxs[global_input_idx];
    } else {
      // Fill with infinity
      padded_vals[global_padded_idx] = 2147483647; // INT_MAX
      padded_idxs[global_padded_idx] = 0;          // Dummy value
    }
  }
}

__global__ void batched_bitonic_step_kernel(int *values, unsigned int *indices,
                                            int j, int k, int padded_len,
                                            int total_blocks) {
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  // Robust implementation for Global Memory:
  // We execute one thread per element, but only operate if (i < ixj).
  // This wastes half the threads but simplifies massive indexing.
  // MAPPING: "1 thread = 1 element" strategy with filtering.

  unsigned int i = tid;
  if (i >= total_blocks * padded_len)
    return;

  int my_block = i / padded_len;
  int i_local_real = i % padded_len;

  // Calculate the partner index locally within the block
  int ixj_local = i_local_real ^ j;

  // If the partner is outside my block (should not happen if j < padded_len),
  // abort
  if (ixj_local >= padded_len)
    return;

  // Convert local partner index back to global index
  int ixj = my_block * padded_len + ixj_local;

  // Avoid duplication and ensure we only swap once per pair
  if (ixj > i) {
    int v1 = values[i];
    int v2 = values[ixj];

    bool ascending = ((i_local_real & k) == 0);

    if ((ascending && v1 > v2) || (!ascending && v1 < v2)) {
      // Swap Values
      values[i] = v2;
      values[ixj] = v1;

      // Swap Indices
      unsigned int tmp = indices[i];
      indices[i] = indices[ixj];
      indices[ixj] = tmp;
    }
  }
}

__global__ void copy_from_padded_kernel(const int *padded_vals,
                                        const unsigned int *padded_idxs,
                                        unsigned int *output_idxs,
                                        int valid_len, int padded_len,
                                        int total_blocks) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < total_blocks * valid_len) {
    int block_id = tid / valid_len;
    int offset_in_block = tid % valid_len;

    // Valid data (not infinity) will be at the beginning of each padded block
    // because we sorted in ascending order.
    int global_padded_idx = block_id * padded_len + offset_in_block;
    int global_output_idx = tid;

    output_idxs[global_output_idx] = padded_idxs[global_padded_idx];
  }
}

// --- BATCHED SORT IMPLEMENTATION ---

/**
 * @brief Host function orchestrating the Batched Bitonic Sort.
 */
void batched_gpu_argsort(unsigned short *d_keys, unsigned int *d_indices,
                         size_t num_blocks, size_t block_len) {

  int padded_len = next_power_of_2((int)block_len);
  size_t total_padded_elements = num_blocks * padded_len;

  // 1. Allocate Temporary Padded Buffers
  int *d_padded_keys;
  unsigned int *d_padded_indices;

  CHECK_CUDA_ERROR(
      cudaMalloc(&d_padded_keys, total_padded_elements * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_padded_indices,
                              total_padded_elements * sizeof(unsigned int)));

  // 2. Copy and Pad (Input -> Padded Temp)
  int threads = 256;
  int blocks = (total_padded_elements + threads - 1) / threads;

  copy_to_padded_kernel<<<blocks, threads>>>(d_keys, d_indices, d_padded_keys,
                                             d_padded_indices, block_len,
                                             padded_len, num_blocks);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 3. Batched Bitonic Sort Loop
  // Note: Launching threads to cover all PADDED elements
  blocks = (total_padded_elements + threads - 1) / threads;

  for (int k = 2; k <= padded_len; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      batched_bitonic_step_kernel<<<blocks, threads>>>(
          d_padded_keys, d_padded_indices, j, k, padded_len, num_blocks);
    }
  }
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 4. Copy Back (Padded Temp -> Output)
  // We only need threads for the real elements
  int total_real_elements = num_blocks * block_len;
  blocks = (total_real_elements + threads - 1) / threads;

  copy_from_padded_kernel<<<blocks, threads>>>(
      d_padded_keys, d_padded_indices,
      d_indices, // Write final result here
      block_len, padded_len, num_blocks);
  CHECK_CUDA_ERROR(cudaGetLastError()); // Check errors

  // 5. Cleanup
  CHECK_CUDA_ERROR(cudaFree(d_padded_keys));
  CHECK_CUDA_ERROR(cudaFree(d_padded_indices));
}