<!-- Copilot Instructions for the TFM image-cipher repository -->
# Codebase Orientation for AI Coding Agents

This file contains concise, actionable guidance to help AI coding agents be productive in this repository.

- **Big picture:** The project implements an image encryption pipeline that mixes chaotic maps and elementary cellular automata. High-performance CUDA kernels (C++) perform the heavy-lifting; Python provides convenience scripts and PyCUDA front-ends for experiments and visualization.

- **Where the core logic lives:**
  - `c++/src/` — CUDA/C++ implementation and the CLI entrypoint: `main.cu`.
  - `c++/include/` — public headers (look for `encryption.cuh`, `structs.cuh`, `kernels.cuh`).
  - `python/` — Python tools, experiments, and PyCUDA interfaces (see `encrypt_image.py`, `proposed_cipher_cuda.py`).
  - `repositorio/` — sample image datasets used during development.

- **Build & run (C++/CUDA):**
  - Build (normal): `cd c++ && make` (there is a VS Code task: "Build Normal").
  - Build (debug): `cd c++ && make MODE=debug` (adds `-G -g -O0`).
  - Binary: `c++/bin/cipher.out` (created by the Makefile).
  - Quick runner: `c++/compile_and_execute.bash` demonstrates a full encrypt+decrypt run.

- **CLI usage (important — fixed arg order):**
  - Signature (exact 10 numeric args expected after paths/password):
    `./bin/cipher.out <InputPath> <OutputPath> <Password> <Rounds> <Verbose(0/1)> <Mode(1=Enc/0=Dec)> <BlockSize> <Precision> <AutoSteps> <TransLen>`
  - `main.cu` enforces `required_args == 11` (program name + 10 arguments). Keep this ordering and types.
  - Example encrypt (from `compile_and_execute.bash`):
    `./bin/cipher.out ../repositorio/set3/lena3.jpg ./bin/salida.tif password 3 1 1 8 2 20 10`

- **Key data structures & semantics:**
  - `EncryptionParams` (see `c++/include/structs.cuh`): `rounds`, `block_size`, `precision_level`, `automata_steps`, `transition_length` — these control algorithm phases.
  - Image handling: code uses `cv::imread(..., IMREAD_UNCHANGED)` and explicitly unstacks/stacks RGB channels inside `encrypt_image` (see `encryption.cuh`). Don’t change channel handling without verifying restacking logic.

- **CUDA & toolchain notes:**
  - `Makefile` uses `nvcc` and explicitly sets `-ccbin g++-12`. Ensure `g++-12` and a compatible CUDA toolkit are available when building.
  - Linking: OpenCV via `pkg-config --libs opencv4`, plus `-lssl -lcrypto -lcudart`.

- **Python environment:**
  - `python/requirements.txt` lists exact versions (PyCUDA, CuPy). Use `pip install -r python/requirements.txt` in a Python 3.8+ environment with CUDA installed.
  - Python scripts call PyCUDA/CuPy and expect an NVIDIA GPU + matching CUDA drivers.

- **Project-specific conventions for contributors/agents:**
  - Preserve the CLI argument contract; unit tests and external scripts rely on that exact order and counts.
  - Use `c++/src` + `c++/include` pair when editing CUDA kernels and expose new helpers via header prototypes in `include/`.
  - When changing kernel launches or memory layout, test both color and grayscale images (code paths diverge on channels).
  - Avoid changing NVCC flags (`-rdc=true`) or `-ccbin` without running the full pipeline — this repo depends on device linking and specific toolchain assumptions.

- **Where to look for behavior examples/tests:**
  - `c++/compile_and_execute.bash` — end-to-end example run.
  - `python/` scripts like `encrypt_image.py`, `unencrypt_image.py`, and `TestNIST.py` for statistic checks and evaluation utilities.

- **When proposing changes, include:**
  - A short rationale (security, performance), the limited code diff, and new test commands to validate both encrypt and decrypt paths using `repositorio/set3/lena3.tif` or the example runner.

If anything here is unclear or you want a different level of detail (e.g., annotated call graph or a test harness), tell me which area to expand. 
