# Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Parameter Tuning](#parameter-tuning)
5. [Common Use Cases](#common-use-cases)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Getting Started

### Prerequisites

- **CUDA-capable NVIDIA GPU** with compute capability 3.0+
- **CUDA Toolkit** (tested with CUDA 11.0+)
- **g++-12** or compatible C++ compiler
- **OpenCV 4.x** with development headers
- **OpenSSL** development libraries

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd TFM-ALGORITMO-CRIPTOGRAFICO-PARA-EL-CIFRADO-DE-IMAGENES
   ```

2. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential g++-12
   sudo apt-get install libopencv-dev libssl-dev
   sudo apt-get install nvidia-cuda-toolkit  # If not already installed
   ```

3. **Build the cipher:**
   ```bash
   cd cuda
   make -j8
   ```

4. **Verify the build:**
   ```bash
   ls -lh bin/cipher.out
   ```

5. **Optional: Install Python analysis tools:**
   ```bash
   cd ../python
   pip install -r requirements.txt
   ```

## Basic Usage

### Encrypting an Image

```bash
cd /path/to/TFM-ALGORITMO-CRIPTOGRAFICO-PARA-EL-CIFRADO-DE-IMAGENES

./cuda/bin/cipher.out \
    ./repositorio/set3/lena3.jpg \
    ./output/encrypted.tif \
    mypassword \
    3 1 8 4 20 10 3.9 1
```

**Arguments explained:**
- `./repositorio/set3/lena3.jpg` - Input image
- `./output/encrypted.tif` - Output encrypted image
- `mypassword` - Your encryption password
- `3` - Number of rounds
- `1` - Mode: 1 = encrypt
- `8` - Block size
- `4` - Precision level
- `20` - Automata steps
- `10` - Transition length
- `3.9` - Chaos parameter
- `1` - Verbose: 1 = show progress

### Decrypting an Image

```bash
./cuda/bin/cipher.out \
    ./output/encrypted.tif \
    ./output/decrypted.jpg \
    mypassword \
    3 0 8 4 20 10 3.9 1
```

**Key difference:** Mode is `0` (decrypt) instead of `1`

### Quick Test (Encrypt + Decrypt)

Use the provided helper script:

```bash
./compile_and_execute.bash 3
```

This will:
1. Build the project
2. Encrypt a test image (3 rounds)
3. Decrypt the result
4. Compare original vs decrypted

## Advanced Usage

### Streaming Mode (STDIN/STDOUT)

Process images through pipes without writing temporary files:

**Example 1: Encrypt and pipe to file**
```bash
cat input.jpg | \
./cuda/bin/cipher.out STDIN STDOUT mypassword 3 1 8 4 20 10 3.9 0 \
> encrypted.tif
```

**Example 2: Decrypt from pipe**
```bash
cat encrypted.tif | \
./cuda/bin/cipher.out STDIN STDOUT mypassword 3 0 8 4 20 10 3.9 0 \
> decrypted.jpg
```

**Example 3: Chain operations**
```bash
# Encrypt and decrypt in one command
cat original.jpg | \
./cuda/bin/cipher.out STDIN STDOUT pass 3 1 8 4 20 10 3.9 0 | \
./cuda/bin/cipher.out STDIN STDOUT pass 3 0 8 4 20 10 3.9 0 \
> recovered.jpg
```

### Display Mode (GUI)

Show the result in a window instead of saving to file:

```bash
./cuda/bin/cipher.out \
    ./repositorio/set3/lena3.jpg \
    SHOW \
    mypassword \
    3 1 8 4 20 10 3.9 1
```

**Note:** Requires X11/display server. Not available in headless environments.

### Batch Processing

Encrypt multiple images:

```bash
for img in ./repositorio/set3/*.jpg; do
    basename=$(basename "$img" .jpg)
    ./cuda/bin/cipher.out \
        "$img" \
        "./output/${basename}_encrypted.tif" \
        mypassword \
        3 1 8 4 20 10 3.9 0
done
```

### Performance Profiling

Build in debug mode and use NVIDIA profiler:

```bash
cd cuda
make clean
make MODE=debug

nvprof ./bin/cipher.out ../repositorio/set3/lena3.jpg output.tif password 3 1 8 4 20 10 3.9 0
```

Or use Nsight Compute:

```bash
ncu --set full ./bin/cipher.out ../repositorio/set3/lena3.jpg output.tif password 3 1 8 4 20 10 3.9 0
```

## Parameter Tuning

### Security vs Performance Trade-off

| Parameter | Low Security (Fast) | Balanced | High Security (Slow) |
|-----------|---------------------|----------|---------------------|
| `Rounds` | 1 | 3 | 5 |
| `BlockSize` | 32 | 8-16 | 8 |
| `Precision` | 2 | 4 | 8 |
| `AutoSteps` | 10 | 20-50 | 100 |
| `TransLen` | 5 | 10-20 | 50 |

### Recommended Configurations

**Fast Encryption (Real-time applications):**
```bash
./cuda/bin/cipher.out input.jpg output.tif password 1 1 32 2 10 5 3.9 0
```

**Balanced (General use):**
```bash
./cuda/bin/cipher.out input.jpg output.tif password 3 1 8 4 20 10 3.9 0
```

**Maximum Security (Sensitive data):**
```bash
./cuda/bin/cipher.out input.jpg output.tif password 5 1 8 8 100 50 3.999 0
```

### Chaos Parameter Selection

The `chaosParam` must be in the chaotic regime of the logistic map:

- **Chaotic range:** 3.57 < r ≤ 4.0
- **Recommended values:**
  - `3.9` - Good balance
  - `3.999` - Maximum chaos
  - `4.0` - Boundary (avoid)

**Verify chaotic behavior:**
```bash
cd python
python bifurcacion.py  # Visualize bifurcation diagram
python lyapunov.py     # Compute Lyapunov exponent
```

### Image Size Considerations

| Image Size | Recommended BlockSize | Memory Usage (approx) |
|------------|----------------------|----------------------|
| 512×512 | 8-16 | <100 MB |
| 1024×1024 | 8-16 | <500 MB |
| 4096×4096 | 16-32 | ~2 GB |
| 8192×8192 | 32 | ~8 GB |

For very large images (>16 MP), consider:
- Increasing `BlockSize` to 32 or 64
- Reducing `Rounds` if performance is critical
- Ensuring sufficient GPU memory

## Common Use Cases

### Use Case 1: Secure Image Storage

Encrypt images before uploading to cloud storage:

```bash
# Encrypt before upload
./cuda/bin/cipher.out local_image.jpg encrypted.tif StrongPassword123! 3 1 8 4 20 10 3.9 0
# Upload encrypted.tif to cloud

# Later: download and decrypt
./cuda/bin/cipher.out encrypted.tif decrypted.jpg StrongPassword123! 3 0 8 4 20 10 3.9 0
```

### Use Case 2: Research & Analysis

Evaluate encryption quality:

```bash
# Run comprehensive cryptographic analysis
cd python
python stats.py ../repositorio/set3/lena3.jpg password ../cuda/bin/cipher.out --rounds 3

# Results in:
# - Console table with metrics
# - full_report.jpg with visualizations
```

### Use Case 3: Batch Processing with Different Keys

```bash
#!/bin/bash
# encrypt_batch.sh

IMAGES_DIR="./repositorio/set3"
OUTPUT_DIR="./encrypted"
KEYS_FILE="keys.txt"  # One key per line

mkdir -p "$OUTPUT_DIR"

i=0
while IFS= read -r key; do
    for img in "$IMAGES_DIR"/*.jpg; do
        basename=$(basename "$img" .jpg)
        ./cuda/bin/cipher.out \
            "$img" \
            "$OUTPUT_DIR/${basename}_${i}.tif" \
            "$key" \
            3 1 8 4 20 10 3.9 0
    done
    ((i++))
done < "$KEYS_FILE"
```

### Use Case 4: Video Frame Encryption

Encrypt video frames individually:

```bash
# Extract frames
ffmpeg -i video.mp4 frames/frame_%04d.jpg

# Encrypt all frames
for frame in frames/*.jpg; do
    ./cuda/bin/cipher.out "$frame" "encrypted/$(basename $frame .jpg).tif" password 2 1 16 4 20 10 3.9 0
done

# Reassemble (would need to decrypt first for viewing)
```

## Troubleshooting

### Build Issues

**Problem:** `nvcc: command not found`

**Solution:**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem:** `fatal error: opencv2/core.hpp: No such file or directory`

**Solution:**
```bash
sudo apt-get install libopencv-dev
# Or check pkg-config
pkg-config --cflags --libs opencv4
```

**Problem:** `undefined reference to cv::imread`

**Solution:** Ensure OpenCV is properly linked. Check `Makefile` LDFLAGS.

### Runtime Issues

**Problem:** `[ERROR] Image data is empty or corrupted`

**Solutions:**
- Verify input file exists and is a valid image
- Check file permissions
- Try a different image format (JPEG, PNG, TIFF all supported)

**Problem:** `CUDA error: out of memory`

**Solutions:**
- Reduce image size
- Increase `BlockSize` parameter
- Close other GPU applications
- Check available GPU memory: `nvidia-smi`

**Problem:** Decrypted image doesn't match original

**Causes:**
- Incorrect password
- Different parameters used for encryption/decryption
- Parameters must match exactly (rounds, blocksize, precision, etc.)

**Problem:** `[ERROR] GUI Call failed (No X11/Display?)`

**Solution:**
- Use file output instead of `SHOW` mode
- Or enable X11 forwarding: `ssh -X` or use VNC

### Performance Issues

**Problem:** Encryption is very slow

**Solutions:**
- Reduce `Rounds` (try 1-2 instead of 5)
- Increase `BlockSize` (try 32 instead of 8)
- Reduce `AutoSteps` and `TransLen`
- Disable verbose mode (set last argument to 0)
- Check GPU utilization: `nvidia-smi`

**Problem:** GPU is underutilized

**Solutions:**
- Ensure CUDA toolkit version matches GPU architecture
- Try larger images (more parallelism)
- Check no CPU fallback is occurring

## FAQ

**Q: What image formats are supported?**

A: All formats supported by OpenCV: JPEG, PNG, TIFF, BMP, WebP, etc. TIFF is recommended for lossless encrypted storage.

**Q: Can I encrypt the same image multiple times?**

A: Yes! Each encryption with the same parameters produces identical output (deterministic). Use different passwords or parameters for different encryption operations.

**Q: How secure is this encryption?**

A: The algorithm provides strong confusion and diffusion. Security analysis (NPCR, UACI, Entropy) shows it meets standard cryptographic criteria. However, domain-specific encryption is generally not a replacement for AES for general-purpose encryption.

**Q: Can I use this for real-world security applications?**

A: This is a research/academic implementation. For production use:
- Conduct a thorough security audit
- Add authenticated encryption (MAC)
- Implement secure key management
- Consider using standard ciphers (AES-GCM) for critical applications

**Q: What's the maximum image size?**

A: Limited by GPU memory. Most modern GPUs can handle 8K images (8192×8192). For larger images, use tiling or increase BlockSize.

**Q: Can I decrypt on a different machine?**

A: Yes, as long as:
- Same password
- Same parameters (rounds, blockSize, precision, autoSteps, transLen, chaosParam)
- CUDA-capable GPU (decryption also uses GPU)

**Q: Why use TIFF for encrypted output?**

A: TIFF is lossless. Encrypting produces high-entropy data that looks random. Lossy formats (JPEG) will corrupt this data and prevent decryption.

**Q: How do I choose a good password?**

A: Use a strong password:
- At least 12 characters
- Mix of letters, numbers, symbols
- Avoid dictionary words
- Consider using a passphrase

**Q: What happens if I lose the password?**

A: The image cannot be recovered. There is no password recovery mechanism. Back up your passwords securely.

**Q: Can I modify the code?**

A: Yes, the code is provided for research and educational purposes. See the source files in `cuda/src/` and `cuda/include/`.

**Q: How can I cite this work?**

A: Please refer to the associated Master's Thesis:
> Design and Implementation of a Cryptographic Algorithm for Image Encryption Based on Chaotic Models and Cellular Automata

**Q: Where can I get help?**

A: 
- Check the documentation in this repository
- Review the architecture documentation (ARCHITECTURE.md)
- Examine the source code comments
- Open an issue on the repository (if applicable)
