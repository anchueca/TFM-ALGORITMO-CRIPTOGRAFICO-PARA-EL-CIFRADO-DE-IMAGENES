# Architecture Overview

## System Design

This cryptographic image encryption system combines **chaotic maps** and **elementary cellular automata** to create a secure, GPU-accelerated image cipher. The implementation leverages CUDA for high-performance parallel processing.

## High-Level Architecture

```
┌─────────────┐
│ Input Image │
└──────┬──────┘
       │
       ▼
┌─────────────────┐       ┌──────────────────┐
│ Pre-Processing  │◄──────┤   Password       │
│ (RGB Unstack)   │       │   (Key Material) │
└────────┬────────┘       └────────┬─────────┘
         │                         │
         │                         ▼
         │              ┌──────────────────────┐
         │              │ Key Derivation       │
         │              │ - Automata Seeds     │
         │              │ - Flow Seeds         │
         │              │ - Chaos Parameters   │
         │              └──────────┬───────────┘
         │                         │
         │                         ▼
         │              ┌──────────────────────┐
         │              │ Permutation          │
         │              │ Generation (GPU)     │
         │              │ - Row Permutations   │
         │              │ - Col Permutations   │
         │              │ - Block Permutations │
         │              └──────────┬───────────┘
         │                         │
         ▼                         ▼
┌──────────────────────────────────────────┐
│      Encryption Pipeline (GPU)           │
│  ┌────────────────────────────────────┐  │
│  │ For each round:                    │  │
│  │  1. Confusion (Substitution)       │  │
│  │     - XOR with chaotic flow        │  │
│  │  2. Diffusion (Permutation)        │  │
│  │     - Row permutation              │  │
│  │     - Column permutation           │  │
│  │     - Block permutation (2 passes) │  │
│  └────────────────────────────────────┘  │
└──────────────────┬───────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ Post-Processing │
         │ (RGB Restack)   │
         └────────┬────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Encrypted Image│
         └────────────────┘
```

## Core Components

### 1. Key Derivation System

**Location:** `cuda/include/encryption_aux.cuh`, `cuda/src/encryption_aux.cu`

The password is transformed into multiple key streams used for different permutation types:

- **Password Segmentation:** The password is hashed and expanded into segments
- **Automata Seeds:** Used to initialize cellular automata states
- **Flow Seeds:** Used to generate chaotic sequences
- **Chaos Parameters:** Control parameters for the logistic map

**Key Functions:**
- `calculate_password()`: Derives key material from the user password
- `convert_bits_to_real()`: Converts binary key material to floating-point seeds

### 2. Permutation Generators

#### Elementary Cellular Automata (ECA)

**Location:** `cuda/include/automata.cuh`, `cuda/src/automata.cu`

Implements 1D elementary cellular automata to generate pseudorandom permutations.

**Key Features:**
- GPU-accelerated evolution using shared memory
- Configurable rules (0-255)
- Periodic boundary conditions
- Bit-packed state representation for memory efficiency

**Usage:** Generates row and column permutations

**Class:** `ElementalCelularAutomata`
- Manages automaton state on GPU
- Evolves state for N steps
- Extracts permutation sequences

#### Chaotic Flow Permutations

**Location:** `cuda/include/encryption_aux.cuh`

Uses the **logistic map** to generate chaotic sequences:

$$x_{n+1} = r \cdot x_n \cdot (1 - x_n)$$

Where $r$ is the chaos parameter (typically 3.57-4.0 for chaotic behavior).

**Function:** `generate_flow_permutations()`
- Creates permutations from chaotic trajectories
- Uses multiple passes for enhanced randomness
- Applied to block-level permutations

### 3. Encryption Pipeline

**Location:** `cuda/include/encryption.cuh`, `cuda/src/encryption.cu`

#### Main Orchestrator: `encrypt_image()`

Coordinates the full encryption/decryption process:
1. Pre-processing (RGB unstacking for color images)
2. GPU memory allocation
3. Key generation
4. Permutation generation
5. Encryption process execution
6. Result retrieval
7. Post-processing (RGB restacking)
8. Cleanup

#### Encryption Process: `encryption_process()`

Each round consists of:

1. **Confusion Phase:**
   - XOR pixel values with chaotic flow sequence
   - Breaks statistical patterns

2. **Diffusion Phase (Image Permutation):**
   - Row permutation: Shuffles rows
   - Column permutation: Shuffles columns
   - Block permutation: Divides image into blocks and shuffles
     - Applied twice for enhanced mixing

**Rounds:** Multiple rounds compound the confusion and diffusion effects

#### Decryption Process: `unencryption_process()`

Reverses the encryption process:
- Inverse block permutation (2 passes)
- Inverse column permutation
- Inverse row permutation
- XOR with chaotic flow (self-inverse)

### 4. GPU Memory Management

**Struct:** `D_pointers` (defined in `structs.cuh`)

Manages all device (GPU) memory pointers:
- `d_image`: Current image buffer
- `d_image_out`: Output image buffer
- `d_flow`: Chaotic flow sequence
- `d_seeds`: Random seeds
- `d_permutation_rows/cols/blocks`: Forward permutations
- `d_permutation_rows/cols/blocks_inverse`: Inverse permutations

**Memory Strategy:**
- Double buffering for image data
- Swap pointers between rounds (zero-copy)
- Pre-computed inverse permutations for decryption

### 5. CUDA Kernels

**Location:** `cuda/include/kernels.cuh`, `cuda/src/kernels.cu`

Low-level GPU kernels for pixel and block operations:
- `confuse_kernel()`: XOR with chaotic flow
- `permute_rows_kernel()`: Row shuffling
- `permute_cols_kernel()`: Column shuffling
- `permute_blocks_kernel()`: Block-level shuffling

**Optimization Techniques:**
- Coalesced memory access
- Shared memory for neighbor access
- Optimal thread/block configuration

## Data Flow

### Encryption Flow

```
Original Image
    ↓
[RGB Split] (if color)
    ↓
GPU Upload (d_image)
    ↓
[Round 1]
 ├─ XOR with Flow
 ├─ Row Permutation
 ├─ Column Permutation
 └─ Block Permutation (×2)
    ↓
[Round 2]
 ├─ ...
    ↓
[Round N]
    ↓
GPU Download
    ↓
[RGB Merge] (if color)
    ↓
Encrypted Image
```

### Decryption Flow

```
Encrypted Image
    ↓
[RGB Split] (if color)
    ↓
GPU Upload (d_image)
    ↓
[Round N] (reversed)
 ├─ Inverse Block Perm (×2)
 ├─ Inverse Col Perm
 ├─ Inverse Row Perm
 └─ XOR with Flow
    ↓
[Round N-1]
 ├─ ...
    ↓
[Round 1]
    ↓
GPU Download
    ↓
[RGB Merge] (if color)
    ↓
Decrypted Image
```

## Security Design

### Confusion (Substitution)

- **XOR with Chaotic Flow:** Each pixel is XORed with a value from a chaotic sequence
- **Non-linearity:** The logistic map provides non-linear mixing
- **Key-dependent:** Flow is generated from password-derived seeds

### Diffusion (Permutation)

- **Multi-level Permutation:**
  - Row: Spreads changes vertically
  - Column: Spreads changes horizontally
  - Block: Spreads changes across regions

- **Cellular Automata:** Provides complex, deterministic permutations
- **Multiple Rounds:** Amplifies the avalanche effect

### Avalanche Effect

Small changes in plaintext or key produce dramatic changes in ciphertext:
- 1-bit change in plaintext → ~50% pixels change (NPCR > 99.6%)
- 1-bit change in key → completely different ciphertext

## Performance Characteristics

### Parallelization

- **Pixel-level Operations:** XOR operations are fully parallel
- **Permutations:** Block and row/column permutations leverage GPU parallelism
- **Automata Evolution:** Each cell is processed by separate threads

### Memory Access Patterns

- **Coalesced Access:** Where possible, threads access contiguous memory
- **Shared Memory:** Used in automata kernel for neighbor access
- **Double Buffering:** Eliminates need for synchronization between rounds

### Scalability

Performance scales with:
- Image size (more pixels = more parallelism)
- GPU compute capability
- Available GPU memory

Typical throughput: **10-100 MB/s** depending on parameters and GPU

## Configuration Parameters

Defined in `EncryptionParams` struct:

- **`rounds`:** Number of encryption cycles (1-5 typical)
  - More rounds = higher security, slower performance
  
- **`block_size`:** Block dimension for block permutations (8, 16, 32 typical)
  - Larger blocks = faster but coarser mixing
  
- **`automata_steps`:** Cellular automata evolution steps (20-100 typical)
  - More steps = more random permutations
  
- **`transition_length`:** Flow permutation passes (10-50 typical)
  - More transitions = better mixing
  
- **`chaos_parameter`:** Logistic map parameter (3.57-4.0)
  - Must be in chaotic regime (typically 3.9 or 3.999)

## File Organization

```
cuda/
├── include/           # Header files
│   ├── encryption.cuh      # Main orchestration
│   ├── automata.cuh        # Cellular automata
│   ├── encryption_aux.cuh  # Key derivation & helpers
│   ├── kernels.cuh         # GPU kernel declarations
│   ├── kernels_aux.cuh     # Kernel helpers
│   ├── CudaPermutation.cuh # Permutation utilities
│   ├── aux.cuh             # General utilities
│   └── structs.cuh         # Data structures
└── src/               # Implementation files
    ├── main.cu             # CLI entry point
    ├── encryption.cu       # Main encryption logic
    ├── automata.cu         # Automata implementation
    ├── encryption_aux.cu   # Key derivation impl
    ├── kernels.cu          # GPU kernels
    ├── kernels_aux.cu      # Kernel helper impl
    ├── CudaPermutation.cu  # Permutation impl
    └── aux.cu              # General utilities impl
```

## Integration Points

### CLI Interface

**File:** `cuda/src/main.cu`

Provides command-line interface with:
- File I/O mode: Read/write image files
- Streaming mode: STDIN/STDOUT for pipeline integration
- Display mode: Show results in GUI window

### Python Analysis Tools

**Directory:** `python/`

Interfaces with the cipher binary via subprocess:
- Encrypts/decrypts via STDIN/STDOUT
- Computes cryptographic metrics
- Generates analysis reports

## Future Extensions

Potential areas for enhancement:
- **Multiple Chaotic Maps:** Incorporate other maps (tent, Hénon, etc.)
- **Adaptive Parameters:** Auto-tune based on image characteristics
- **Stream Cipher Mode:** Support for video encryption
- **Multi-GPU:** Distribute across multiple GPUs for very large images
- **Authenticated Encryption:** Add MAC for integrity verification
