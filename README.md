# TFM - Cryptographic Algorithm for Image Encryption

This repository contains the source code developed for the Master's Thesis:

**Design and Implementation of a Cryptographic Algorithm for Image Encryption Based on Chaotic Models and Cellular Automata**

## Description

The algorithm implements an image encryption scheme using:

- **Chaotic dynamics** (cosine-cosine map): generates deterministic pseudo-random sequences
- **Elementary cellular automata** (Rule 30): provides diffusion and permutation scheduling  
- **Block-based permutations** (rows, columns, blocks) combined with pixel diffusion
- **GPU acceleration via CUDA**: optimized kernels for high-performance encryption/decryption

## Overview

The primary implementation is a high-performance C++/CUDA implementation located in the `cuda/` directory. The project focuses on a block-based image cipher combining chaotic maps and elementary cellular automata, implemented and optimized for execution on NVIDIA GPUs.

**Key components:**
- **Encryption engine:** GPU-accelerated CUDA kernels with constant memory optimizations
- **Steganography module:** Chaos-based LSB steganography with lossless recovery using EXIF metadata
- **Cryptographic analysis tools:** Python utilities for statistical testing and performance evaluation
- **CLI interface:** Command-line tool with flexible input/output modes (file, STDIN/STDOUT, display)


## Where the core logic lives

- `cuda/src/` — CUDA/C++ source files (kernels and host code).
- `cuda/include/` — public headers and data structures (e.g. `encryption.cuh`, `structs.cuh`).
- The CLI entrypoint: `cuda/src/main.cu` (build produces `bin/cipher.out`).

## Build

Build the C++/CUDA project from the repository root:

```bash
cd cuda
make
```

Binary: `cuda/bin/cipher.out` (built by the included `Makefile`).

Notes:
- The Makefile uses `nvcc` and sets `-ccbin g++-12`. Ensure `g++-12` and a compatible CUDA toolkit are available.
- OpenCV and OpenSSL are required for image I/O and cryptographic helpers; these are linked via the Makefile.

## CLI Usage

The CLI expects exactly 12 arguments after the binary name (see `main.cu`):

```
./bin/cipher.out <InputPath> <OutputPath|SHOW|STDOUT> <Password> <Rounds> <Mode(1=Enc/0=Dec)> <BlockSize> <AutoSteps> <TransLen> <chaosParam> <Verbose(0/1)>
```

**Arguments:**
- `InputPath`: Path to input image file, or `STDIN` to read from standard input
- `OutputPath`: Path to output file, `SHOW` to display in window (requires GUI), or `STDOUT` to pipe to standard output
- `Password`: Encryption/decryption password
- `Rounds`: Number of encryption rounds (typically 1-5)
- `Mode`: `1` for encryption, `0` for decryption
- `BlockSize`: Size of square blocks for block permutations (e.g., 8, 16, 32)
- `AutoSteps`: Number of cellular automata evolution steps (e.g., 20, 50, 100)
- `TransLen`: Transition sequence length for flow permutations (e.g., 10, 20, 50)
- `chaosParam`: Chaos parameter for the logistic map (typically 3.57-4.0, e.g., 3.9 or 3.999)
- `Verbose`: `1` to enable verbose logging, `0` for silent operation

**Example - Encrypt:**

```bash
./bin/cipher.out ../repositorio/set3/lena3.jpg ./bin/salida.tif password 3 1 8 4 20 10 3.9 1
```

**Example - Decrypt:**

```bash
./bin/cipher.out ./bin/salida.tif ./bin/decrypted.tif password 3 0 8 4 20 10 3.9 1
```

**Example - Streaming mode (STDIN/STDOUT):**

```bash
cat input.jpg | ./bin/cipher.out STDIN STDOUT password 3 1 8 4 20 10 3.9 0 > encrypted.tif
```

The order and count of arguments is strictly enforced by `main.cu`. Other scripts in the repository may assume the same ordering.

## Testing / Example run

A small helper `compile_and_execute.bash` demonstrates a full encrypt+decrypt run using the compiled binary.

## Performance Optimizations

The CUDA implementation includes several optimizations to ensure high throughput and accurate benchmarking:

- **Constant Memory:** Block permutation tables are stored in GPU Constant Memory (`__constant__`) to minimize memory latency during the permutation phase.
- **Initialization Overhead:** A dummy `cudaFree(0)` call is performed before timing starts to absorb the CUDA context initialization cost (~200ms), ensuring that reported metrics reflect the actual algorithm performance (~5-20ms).
- **Accurate Timing:** The C++ binary reports precise execution times (in ms) to `stderr`, which are parsed by the Python analysis tools to exclude invalid process startup overheads.

## Steganography Module

The project includes a **chaos-based steganography module** with lossless recovery capabilities:

### Features

- **Same Chaotic Function:** Uses the identical cosine-cosine map (`chaotic_functio`) as the encryption engine for consistency
- **LSB Embedding:** Hides messages in the least significant bits of image pixels using chaotic position generation
- **EXIF Metadata Storage:** Recovery information is automatically stored in EXIF `UserComment` tag (0x8298) for lossless restoration
- **Deterministic Positioning:** Message positions are derived from a chaotic sequence seeded by the encryption password

### API Usage

```cpp
#include "steganography.hpp"

// Embed message with automatic EXIF metadata storage
std::vector<bool> recovery = embed_message_caos_with_exif(
    image, message_bits, key_bits, "output.jpg");

// Extract message with automatic EXIF recovery info reading
std::vector<bool> extracted = extract_message_caos_with_exif(
    image, key_bits, "input.jpg");
```

### How It Works

1. **Embedding Phase:**
   - Generate chaotic sequence using `chaotic_cosine(x, r=2.5)` with the password-derived key
   - Calculate LSB positions based on chaotic values
   - Replace LSBs with message bits
   - Generate recovery information: `R[i] = original_LSB XOR message_bit`
   - **Automatically save recovery info to EXIF metadata**

2. **Extraction Phase:**
   - **Read recovery information from EXIF metadata**
   - Regenerate the same chaotic sequence using the password
   - Extract bits from the same positions
   - Restore original LSBs using recovery information: `original = current_LSB XOR recovery_bit`
   - Return hidden message

### Dependencies

```bash
# Required for EXIF metadata support
apt-get install libexif-dev
```

The Makefile automatically includes libexif during compilation.

## Contact / Notes

- GPU support is required. Ensure a compatible NVIDIA GPU and CUDA drivers are installed.
- Keep the CLI argument contract when integrating other tools or wrappers.


## Image Quality and Cryptographic Analysis

This project includes standard image-quality and cryptographic-statistics that help evaluate the effectiveness and robustness of the cipher. The tests below are commonly used in the literature for image ciphers; brief definitions, formulas, and example commands/snippets are provided to reproduce results locally.

- **Bit Change Rate (BCR):** Measures the percentage of bits that differ when a single bit of the plaintext (or key) is flipped. Higher values indicate strong avalanche behaviour.

- **NPCR (Number of Pixel Change Rate):** Measures the percentage of pixels that change between two cipher-images (e.g., original vs. one-bit-changed). Formula:

	$$\text{NPCR} = \frac{\sum_{i,j} D(i,j)}{M\times N}\times 100\%$$

	where
	- $D(i,j)=1$ if $C_1(i,j) \ne C_2(i,j)$, otherwise $D(i,j)=0$;
	- $M,N$ are image dimensions (multiply by number of channels for color images).

- **UACI (Unified Average Changing Intensity):** Measures average intensity differences between two ciphertexts. Formula:

	$$\text{UACI} = \frac{1}{M\times N}\sum_{i,j} \frac{|C_1(i,j)-C_2(i,j)|}{L-1}\times 100\%$$

	where $L$ is the number of grey levels (usually 256).

- **Correlation Coefficient (CC):** Measures the correlation between adjacent pixels (horizontal, vertical, diagonal) in the encrypted image. A secure cipher should produce coefficients near 0.

- **Information Entropy (IE):** Measures randomness of the pixel value distribution. For an 8-bit grayscale channel:

	$$IE = -\sum_{v=0}^{255} p(v)\log_2 p(v)$$

	Values close to 8 indicate high randomness for an 8-bit image.

- **MSE / PSNR:** Standard image-quality metrics useful to quantify distortion when comparing original vs decrypted images. PSNR is derived from MSE.

- **Chi-Square Test:** Assesses the uniformity of the pixel value distribution in the ciphertext; lower p-values indicate non-uniformity.

- **Key Sensitivity Test:** Encrypt the same plaintext with two keys differing by one bit and measure NPCR/UACI to assess sensitivity to key changes.

Practical example: generate two ciphertexts and compute NPCR/UACI using Python + OpenCV + NumPy (quick one-shot):

```bash
# encrypt original
./cuda/bin/cipher.out ../repositorio/set3/lena3.jpg ./tmp/ct1.tif password 3 1 8 4 20 10 3.9 0
# flip one bit in the plaintext or use a different key to produce ct2 (example using the same binary with a slightly different password)
./cuda/bin/cipher.out ../repositorio/set3/lena3.jpg ./tmp/ct2.tif password_alt 3 1 8 4 20 10 3.9 0

python - <<'PY'
import cv2, numpy as np
a = cv2.imread('tmp/ct1.tif', cv2.IMREAD_UNCHANGED)
b = cv2.imread('tmp/ct2.tif', cv2.IMREAD_UNCHANGED)
channels = a.shape[2] if a.ndim==3 else 1
M, N = a.shape[0], a.shape[1]
diff = (a != b).astype(np.uint8)
NPCR = 100.0 * np.sum(diff) / (M * N * channels)
UACI = 100.0 * np.sum(np.abs(a.astype(np.int32) - b.astype(np.int32))) / ((255.0) * M * N * channels)
print(f'NPCR: {NPCR:.4f}%  UACI: {UACI:.4f}%')
PY
```

Notes and recommendations:

- Use representative images from `repositorio/` (e.g. `set3/lena3.jpg`) for reproducible comparisons.
- For key-sensitivity tests, change exactly one bit in the key (or use two keys that differ by a single character/bit) and compute NPCR/UACI.
- For plaintext-sensitivity tests, flip a single pixel or single bit in the plaintext and re-encrypt.
- When reporting entropy, compute the per-channel entropy for color images and report the average and per-channel values.

If you would like, I can add a small, self-contained CLI utility (C++ or Python) to compute the standard NPCR/UACI/entropy/CC metrics automatically from two images and produce a concise report.

