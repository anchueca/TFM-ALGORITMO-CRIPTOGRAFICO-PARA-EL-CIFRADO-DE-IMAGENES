# TFM - Cryptographic Algorithm for Image Encryption

This repository contains the source code developed for the Master's Thesis:

**Design and Implementation of a Cryptographic Algorithm for Image Encryption Based on Chaotic Models and Cellular Automata**

## Description

The algorithm implements a image encryption scheme using:

- Chaotic dynamics (cosine-cosine map).
- Elementary cellular automata (rule 30).
- GPU-based parallelization via CUDA (PyCUDA).

The goal is to enhance both security and performance through nonlinear cryptographic mechanisms and highly parallelizable structures executed on GPU architectures.

## Features

- Block-based permutation + pixel diffusion via chaotic flow
- Key generation from passwords using SHA-512
- Row, column, and block permutation based on cellular automata and chaotic values
- XOR encryption using chaotic stream cipher for intensity scrambling
- Efficient implementation using Python + PyCUDA

## Requirements

- Python 3.8+
- PyCUDA
- NumPy
- OpenCV
- Matplotlib

Recommended installation:
```bash
pip install -r requirements.txt
```
## Scripts

Two main scripts are provided to encrypt and decrypt images:

```encrypt_image.py```
Encrypts an image using a password and number of rounds.
**Usage:**
```python encrypt_image.py <input_image> <password> [--rounds <rounds>] --output <output_image> [--show <level>]```
**Arguments:**

- **input_image**: Path to the image to encrypt.

- **password**: Password used to generate the encryption key.

- --**rounds**: Number of encryption rounds.

- --**output**: Path where the encrypted image will be saved.

- --**show**: (Optional) Visualization level (0 = none, 1 = final image, 2+ = debugging steps).

**Example:**
```python encrypt_image.py repositorio/set3/lena3.tif mypassword 3 --output lena_encrypted.tif```

```decrypt_image.py```
Decrypts a previously encrypted image using the same password and rounds.
**Usage:**
```python decrypt_image.py <input_image> <password> [--rounds <rounds>] --output <output_image> [--show <level>]```
**Arguments:**

- **input_image**: Path to the encrypted image.

- **password**: Same password used during encryption.

- --**rounds**: Same number of rounds used in encryption.

- --**output**: Path where the decrypted image will be saved.

- --**show**: (Optional) Visualization level (0 = none, 1 = final image, 2+ = debugging steps).

**Example:**
```python decrypt_image.py lena_encrypted.tif mypassword 3 --output lena_decrypted.tif```


## Image Quality and Cryptographic Analysis

This algorithm includes several statistical tests to evaluate the effectiveness and robustness of the image encryption process. These tests help ensure that the encryption scheme provides good security and preserves image quality.

### Key Tests:
- **Bit Change Rate (BCR)**: Measures the percentage of bits that differ between two encrypted images, indicating how sensitive the cipher is to small image changes.
- **NPCR & UACI**: Evaluate pixel-wise and intensity differences between original and encrypted images.
- **Correlation Coefficient (CC)**: Measures the similarity between original and encrypted images.
- **Entropy (IE)**: Indicates the randomness or unpredictability of pixel values in the encrypted image.
- **Mean Squared Error (MSE) & PSNR**: Standard image quality metrics that measure the distortion introduced by encryption.
- **Chi-Square Test**: Assesses the uniformity of pixel distributions in the encrypted image.
- **Key Sensitivity Test**: Evaluates how small changes in the encryption key affect the ciphertext.

### Example Usage:

```bash
python test_statistics.py --input_image path/to/your/image.png --password "yourpassword"
```


## Notes

- GPU support is required. Make sure your system has a compatible NVIDIA GPU and CUDA toolkit properly installed.

- The same password and number of rounds must be used for both encryption and decryption.

