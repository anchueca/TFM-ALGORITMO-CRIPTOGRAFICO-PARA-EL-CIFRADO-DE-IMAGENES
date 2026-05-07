# Cryptographic Algorithm for Image Encryption

This repository contains the source code developed for the Master's Thesis:

**Design and Implementation of a Cryptographic Algorithm for Image Encryption Based on Chaotic Models and Cellular Automata**

## Description

The algorithm implements a hybrid image encryption scheme combining:

- **Chaotic dynamics** (cosine-cosine map): generate deterministic pseudo-random sequences for diffusion and permutation
- **Elementary cellular automata** (Rule 30): provides permutation scheduling for rows and columns
- **Confusion-diffusion architecture**: block-based permutations (rows, columns, blocks) combined with XOR-based pixel diffusion
- **GPU acceleration via CUDA**: optimized kernels with constant memory for high-performance encryption/decryption
- **Steganography integration**: chaos-based LSB embedding for integrity hash storage

---

## Encryption Scheme Overview

The cipher follows a **confusion-diffusion** paradigm with multiple rounds. The complete encryption pipeline is structured as follows:

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            IMAGE ENCRYPTION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐   ┌─────────────────┐   ┌─────────────────────────────────┐   │
│  │   INPUT     │   │  PREPROCESSING  │   │         KEY DERIVATION          │   │
│  │   IMAGE     │──▶│  Unstack + Pad  │──▶│   Password → SHA-512 → Expand   │   │
│  └─────────────┘   └─────────────────┘   └─────────────────────────────────┘   │
│                                                      │                          │
│                    ┌─────────────────────────────────┴──────────────────────┐   │
│                    ▼                                                        ▼   │
│  ┌─────────────────────────────────┐      ┌─────────────────────────────────┐   │
│  │  PERMUTATION GENERATION         │      │  SEED GENERATION                │   │
│  │  • Rule 30 CA evolution         │      │  • Password segment → Real[]    │   │
│  │  • Argsort → Row/Col perms      │      │  • Seeds for CML keystream      │   │
│  │  • Chaotic values → Block perms │      └─────────────────────────────────┘   │
│  └─────────────────────────────────┘                      │                     │
│                    │                                      │                     │
│                    ▼                                      ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                     INITIAL CONFUSION (×2 iterations)                    │   │
│  │   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐    │   │
│  │   │  Row Permute    │──▶│  Column Permute │──▶│  Block Permute      │    │   │
│  │   │  (CA-based)     │   │  (CA-based)     │   │  (Chaotic-based)    │    │   │
│  │   └─────────────────┘   └─────────────────┘   └─────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                    ┌──────────────────┴──────────────────┐                      │
│                    ▼                                     │                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                 CONFUSION-DIFFUSION ROUNDS (× N rounds)                  │   │
│  │                                                                          │   │
│  │   ┌────────────────────────────────────────────────────────────────┐     │   │
│  │   │  Step A: Generate Chaotic Keystream (CML with coupling)        │     │   │
│  │   │    x_{i,t+1} = (1-ε)·f(x_{i,t}, r) + ε/2·(f(x_{i-1,t}) + ...)  │     │   │
│  │   │    where f(x,r) = cos(π·(r·cos(π·x) - 0.5))   [Cosine-Cosine]  │     │   │
│  │   └────────────────────────────────────────────────────────────────┘     │   │
│  │                              │                                           │   │
│  │   ┌────────────────────────────────────────────────────────────────┐     │   │
│  │   │  Step B: Permute Keystream (2× Row→Col→Block)                  │     │   │
│  │   └────────────────────────────────────────────────────────────────┘     │   │
│  │                              │                                           │   │
│  │   ┌────────────────────────────────────────────────────────────────┐     │   │
│  │   │  Step C: Diffusion - XOR(Image, Permuted_Keystream)            │     │   │
│  │   └────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                    ┌──────────────────┴──────────────────┐                      │
│                    ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                     FINAL CONFUSION (×2 iterations)                      │   │
│  │   Row Permute ──▶ Column Permute ──▶ Block Permute                       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                    ┌──────────────────┴──────────────────┐                      │
│                    ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         HASH EMBEDDING (Steganography)                   │   │
│  │   • Calculate image hash (2-byte checksum)                               │   │
│  │   • Embed in LSBs using chaotic position sequence                        │   │
│  │   • Store recovery info in EXIF metadata                                 │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│                             ┌─────────────────┐                                 │
│                             │  OUTPUT IMAGE   │                                 │
│                             │  (Encrypted)    │                                 │
│                             └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Encryption Process

### 1. Preprocessing Phase

**Channel Unstacking:**
- Color images (RGB) are converted from interleaved format (RGBRGB...) to planar format (RRR...GGG...BBB...)
- This allows treating the image as a single-channel grayscale matrix for uniform processing

**Padding:**
- The image is padded to a square with dimensions divisible by `block_size`
- Original dimensions are stored in the padding region for lossless recovery

### 2. Key Derivation

The user password is processed through a multi-stage key derivation:

```
Password → SHA-512(Password) → Key Expansion → [Segment0, Segment1, Segment2]
```

- **Segment 0**: Used to initialize the Elementary Cellular Automata (Rule 30) for permutation generation
- **Segment 1**: Converted to floating-point seeds for the Coupled Map Lattice (CML) keystream generator
- **Segment 2**: Used for steganography positioning (hash embedding/extraction)

### 3. Permutation Generation

#### 3.1 Row/Column Permutations (CA-Based)

1. **Initialize CA**: Create a Rule 30 elementary cellular automaton with state derived from password segment
2. **Evolution**: Run the automaton for `automata_steps` iterations at block granularity
3. **Value Extraction**: Extract chaotic values from the evolved CA state (groups of 16 bits → unsigned short)
4. **Argsort**: Sort indices by chaotic values to obtain the permutation sequence
5. **Inverse Computation**: Pre-compute inverse permutations for decryption

#### 3.2 Block Permutations (Chaotic-Based)

1. **Transition Period**: Run the CML for `transition_length` iterations to reach chaotic regime
2. **Value Collection**: Collect chaotic values corresponding to `block_size²` positions
3. **Argsort**: Generate block permutation by sorting indices according to chaotic values

### 4. Initial Confusion Phase

Apply permutations to scramble spatial relationships (executed 2× for enhanced diffusion):

```
For j = 0 to 1:
    1. Permute Rows using CA-derived permutation
    2. Permute Columns using CA-derived permutation
    3. Permute Blocks using chaotic-derived permutation
```

### 5. Confusion-Diffusion Rounds

For each round `r` from 1 to `N`:

#### Step A: Chaotic Keystream Generation

The keystream is generated using a **Coupled Map Lattice (CML)** with the **cosine-cosine chaotic map**:

```
f(x, r) = cos(π · (r · cos(π · x) - 0.5))
```

The CML introduces spatial coupling between adjacent cells:

```
x_{i,t+1} = (1-ε) · f(x_{i,t}, r) + (ε/2) · [f(x_{i-1,t}, r) + f(x_{i+1,t}, r)]
```

Where:
- `x_{i,t}` is the state of cell `i` at time `t`
- `r` is the chaos parameter (typically 2.5-3.0 for the cosine-cosine map)
- `ε` is the coupling strength

Additionally, a **Global Seed Mixing** step ensures cross-block diffusion:

```
global_seed_mix(): Update seeds based on mean-field coupling across all blocks
```

#### Step B: Keystream Permutation

Apply the same permutation sequence to the keystream (2× iterations):

```
For j = 0 to 1:
    1. Permute keystream rows
    2. Permute keystream columns
    3. Permute keystream blocks
```

#### Step C: Diffusion (XOR)

Apply bitwise XOR between the image and the permuted keystream:

```
Image[i,j] = Image[i,j] ⊕ Keystream[i,j]
```

This is a symmetric operation: applying XOR twice with the same keystream recovers the original data.

### 6. Final Confusion Phase

Apply identical permutation sequence as the initial confusion (2× iterations):

```
For j = 0 to 1:
    1. Permute Rows
    2. Permute Columns
    3. Permute Blocks
```

### 7. Integrity Hash Embedding

**Purpose**: Store a hash of the original image for tamper detection during decryption.

1. **Hash Calculation**: Compute a 16-bit hash from the original image
2. **Position Generation**: Generate pseudo-random LSB positions using the cosine-cosine chaotic function seeded by password segment 2
3. **Embedding**: Replace LSBs at chaotic positions with hash bits
4. **Recovery Storage**: Store original LSB values in EXIF `UserComment` tag for lossless restoration

---

## Decryption Process

Decryption is the exact inverse of encryption:

1. **Hash Extraction**: Extract embedded hash from LSBs, restore original LSBs from EXIF
2. **Reverse Final Confusion**: Apply **inverse** permutations (Blocks⁻¹ → Columns⁻¹ → Rows⁻¹) ×2
3. **Reverse Diffusion Rounds**: For each round:
   - Regenerate identical keystream (deterministic from password)
   - Apply same permutations to keystream
   - XOR to reverse diffusion (XOR is self-inverse)
4. **Reverse Initial Confusion**: Apply inverse permutations ×2
5. **Postprocessing**: Unpad and restack channels to restore original format

---

## Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| CLI Entry Point | `cuda/src/main.cu` | Command-line interface and orchestration |
| Encryption Core | `cuda/src/encryption.cu` | Main encryption/decryption pipeline |
| Auxiliary Functions | `cuda/src/encryption_aux.cu` | Permutation generation, keystream helpers |
| GPU Kernels | `cuda/src/kernels.cu` | CUDA kernels for parallel operations |
| Cellular Automata | `cuda/src/automata.cu` | Rule 30 CA implementation |
| Steganography | `cuda/src/steganography.cpp` | LSB embedding with EXIF recovery |
| Data Structures | `cuda/include/structs.cuh` | EncryptionParams, D_pointers definitions |

---

## Build

Build the C++/CUDA project from the repository root:

```bash
cd cuda
make
```

Binary: `cuda/bin/cipher.out`

**Requirements:**
- CUDA Toolkit with `nvcc`
- `g++-12` (configured in Makefile)
- OpenCV (image I/O)
- OpenSSL (SHA-512 hashing)
- libexif (EXIF metadata for steganography)

---

## CLI Usage

```
./bin/cipher.out <InputPath> <OutputPath|SHOW|STDOUT> <Password> <Rounds> <Mode(1=Enc/0=Dec)> <BlockSize> <AutoSteps> <TransLen> <chaosParam> <Verbose(0/1)>
```

**Arguments:**

| Argument | Description | Example |
|----------|-------------|---------|
| `InputPath` | Input image or `STDIN` | `input.jpg` |
| `OutputPath` | Output file, `SHOW`, or `STDOUT` | `output.tif` |
| `Password` | Encryption/decryption key | `mypassword` |
| `Rounds` | Confusion-diffusion rounds (1-10) | `3` |
| `Mode` | `1` = encrypt, `0` = decrypt | `1` |
| `BlockSize` | Block permutation size (8, 16, 32) | `8` |
| `AutoSteps` | CA evolution steps (20-100) | `20` |
| `TransLen` | CML transition length (10-50) | `10` |
| `chaosParam` | Chaos parameter r (2.0-3.0) | `2.5` |
| `Verbose` | Enable logging (`1`) or silent (`0`) | `1` |

**Example - Encrypt:**

```bash
./bin/cipher.out input.jpg encrypted.tif password 3 1 8 20 10 2.5 1
```

**Example - Decrypt:**

```bash
./bin/cipher.out encrypted.tif decrypted.jpg password 3 0 8 20 10 2.5 1
```

**Example - Streaming (STDIN/STDOUT):**

```bash
cat input.jpg | ./bin/cipher.out STDIN STDOUT password 3 1 8 20 10 2.5 0 > encrypted.tif
```

---

## Performance Optimizations

- **Constant Memory**: Block permutation tables stored in GPU `__constant__` memory
- **Warm-up**: Dummy `cudaFree(0)` absorbs CUDA context initialization (~200ms overhead)
- **Double Buffering**: Zero-copy buffer swapping via pointer exchange
- **Optimal Thread Configuration**: Dynamic thread/block sizing based on image dimensions
- **Timing Accuracy**: Reports actual algorithm time to `stderr` (excludes startup)

---

## Steganography Module

Chaos-based LSB steganography with lossless recovery:

### Features

- **Same Chaotic Function**: Uses cosine-cosine map consistent with encryption
- **LSB Embedding**: Hides hash bits at chaotic positions
- **EXIF Recovery**: Original LSBs stored in EXIF `UserComment` tag (0x8298)
- **Deterministic**: Reproducible positions from password-derived key

### API

```cpp
#include "steganography.hpp"

// Embed with EXIF storage
embed_message_caos(image, hash_value, key_bits, "output.jpg");

// Extract with EXIF recovery
unsigned short hash = extract_message_caos(image, key_bits, "input.jpg", exif_data);
```

---

## Cryptographic Analysis

Standard metrics for evaluating cipher security:

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **NPCR** | Pixel change rate (1-bit change) | ≈99.6% |
| **UACI** | Average intensity change | ≈33.4% |
| **Correlation** | Adjacent pixel correlation | ≈0 |
| **Entropy** | Information entropy (8-bit) | ≈8.0 |
| **Chi-Square** | Distribution uniformity | Low p-value |

**Quick Test:**

```bash
# Encrypt with two slightly different keys
./cuda/bin/cipher.out input.jpg ct1.tif password 3 1 8 20 10 2.5 0
./cuda/bin/cipher.out input.jpg ct2.tif password_alt 3 1 8 20 10 2.5 0

python3 - <<'PY'
import cv2, numpy as np
a = cv2.imread('ct1.tif', cv2.IMREAD_UNCHANGED)
b = cv2.imread('ct2.tif', cv2.IMREAD_UNCHANGED)
M, N, C = a.shape if a.ndim==3 else (*a.shape, 1)
NPCR = 100.0 * np.sum(a != b) / (M * N * C)
UACI = 100.0 * np.sum(np.abs(a.astype(np.int32) - b)) / (255.0 * M * N * C)
print(f'NPCR: {NPCR:.4f}%  UACI: {UACI:.4f}%')
PY
```

---

## Notes

- **GPU Required**: NVIDIA GPU with compatible CUDA drivers
- **Parameter Consistency**: Encryption and decryption MUST use identical parameters
- **Lossless Format**: Use `.tif` or `.png` for encrypted output to avoid compression artifacts
