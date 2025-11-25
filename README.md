# TFM - Cryptographic Algorithm for Image Encryption

This repository contains the source code developed for the Master's Thesis:

**Design and Implementation of a Cryptographic Algorithm for Image Encryption Based on Chaotic Models and Cellular Automata**

## Description

The algorithm implements a image encryption scheme using:

## Overview

The primary implementation is a high-performance C++/CUDA implementation located in the `cuda/` directory. The project focuses on a image cipher combining chaotic maps and elementary cellular automata, implemented and optimized for execution on NVIDIA GPUs.


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

The CLI expects exactly 10 numeric arguments after the binary name (see `main.cu`):

```
./bin/cipher.out <InputPath> <OutputPath> <Password> <Rounds> <Verbose(0/1)> <Mode(1=Enc/0=Dec)> <BlockSize> <Precision> <AutoSteps> <TransLen>
```

Example encrypt:

```bash
./bin/cipher.out ../repositorio/set3/lena3.jpg ./bin/salida.tif password 3 1 1 8 2 20 10
```

The order and count of arguments is enforced by `main.cu`. Other scripts in the repository may assume the same ordering.

## Testing / Example run

A small helper `compile_and_execute.bash` demonstrates a full encrypt+decrypt run using the compiled binary.


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
./cuda/bin/cipher.out ../repositorio/set3/lena3.jpg ./tmp/ct1.tif password 3 1 1 8 2 20 10
# flip one bit in the plaintext or use a different key to produce ct2 (example using the same binary with a slightly different password)
./cuda/bin/cipher.out ../repositorio/set3/lena3.jpg ./tmp/ct2.tif password_alt 3 1 1 8 2 20 10

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


