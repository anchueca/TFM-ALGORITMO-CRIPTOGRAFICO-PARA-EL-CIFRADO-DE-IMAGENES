# Python Analysis Tools

This directory contains a comprehensive suite of cryptographic analysis and visualization tools for the image encryption algorithm.

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- OpenCV (`opencv-python`)
- NumPy, SciPy, Matplotlib
- scikit-image (for GLCM analysis)
- PyCUDA (for GPU-accelerated analysis)
- Tabulate (for formatted output)

## Main Tools

### stats.py - Comprehensive Cryptographic Analysis

The primary analysis tool that evaluates encryption quality using standard cryptographic metrics.

**Usage:**
```bash
python stats.py <input_image> <password> <cipher_executable> [--rounds N]
```

**Arguments:**
- `input_image`: Path to the plain image to analyze
- `password`: Encryption password
- `cipher_executable`: Path to the compiled cipher binary (`../cuda/bin/cipher.out`)
- `--rounds`: Number of encryption rounds (default: 3)

**Example:**
```bash
python stats.py ../repositorio/set3/lena3.jpg mypassword ../cuda/bin/cipher.out --rounds 3
```

**Metrics Computed:**

1. **Shannon Entropy** - Measures randomness of pixel distribution
   - Ideal value: ~7.999 for 8-bit images
   - Higher values indicate better randomness

2. **Correlation Coefficients** (Horizontal, Vertical, Diagonal)
   - Measures correlation between adjacent pixels
   - Ideal value: ~0.0 (no correlation)

3. **NPCR (Number of Pixel Change Rate)**
   - Differential attack resistance
   - Ideal value: >99.6%

4. **UACI (Unified Average Changing Intensity)**
   - Measures average intensity change
   - Ideal value: ~33.4%

5. **Chi-Square Test**
   - Tests histogram uniformity
   - P-value > 0.05 indicates uniform distribution

6. **GLCM Properties** (Gray-Level Co-occurrence Matrix)
   - Contrast, Homogeneity, Energy
   - Analyzes texture characteristics

7. **DFT Spectrum**
   - Frequency domain analysis
   - Visualizes spectral distribution

8. **Key Sensitivity Test**
   - Tests NPCR/UACI with keys differing by 1 bit

9. **Occlusion Attack**
   - Tests robustness to 25% data loss

10. **Performance Benchmarking**
    - Scalability analysis across multiple image sizes
    - Encryption/decryption timing

**Output:**

Generates `full_report.jpg` - a comprehensive visual dashboard containing:
- Original, encrypted, and decrypted images
- Key sensitivity difference visualization
- Occlusion attack results
- Frequency spectra (DFT)
- Pixel value histograms
- Correlation scatter plots
- Performance scaling charts

Console output displays a formatted table with all metric values.

---

### bifurcacion.py - Bifurcation Diagram Generator

Generates bifurcation diagrams for chaotic map analysis.

**Purpose:** Visualizes the behavior of the logistic map across different parameter values to identify chaotic regimes.

**Usage:**
```bash
python bifurcacion.py
```

This tool helps verify that the chaotic parameter chosen (e.g., 3.9, 3.999) falls within the chaotic regime of the logistic map.

---

### lyapunov.py - Lyapunov Exponent Calculator

Computes Lyapunov exponents to quantify chaotic behavior.

**Purpose:** Positive Lyapunov exponents indicate chaotic behavior and sensitivity to initial conditions.

**Usage:**
```bash
python lyapunov.py
```

This tool validates that the chaotic maps used in key generation exhibit genuine chaotic properties.

---

### Chaos_Generator.py - Chaotic Sequence Generator

Utilities for generating chaotic sequences from various chaotic maps.

**Purpose:** Provides reusable functions for generating pseudorandom sequences from chaotic systems.

---

### TestNIST.py - NIST Statistical Test Suite

Integration with NIST randomness tests.

**Purpose:** Performs standardized statistical tests to evaluate the randomness quality of encrypted images.

**Usage:**
```bash
python TestNIST.py
```

Tests include:
- Frequency tests
- Runs tests
- Spectral tests
- And other NIST SP 800-22 tests

---

### plot.py - Visualization Utilities

General-purpose plotting utilities for analysis results.

**Purpose:** Provides helper functions for creating publication-quality plots and visualizations.

---

## Typical Workflow

1. **Build the cipher:**
   ```bash
   cd ../cuda
   make -j8
   ```

2. **Run comprehensive analysis:**
   ```bash
   cd ../python
   python stats.py ../repositorio/set3/lena3.jpg password123 ../cuda/bin/cipher.out --rounds 3
   ```

3. **Review the results:**
   - Check console output for metric values
   - Open `full_report.jpg` for visual analysis

4. **Optional - Chaos analysis:**
   ```bash
   python bifurcacion.py  # Verify chaotic parameter selection
   python lyapunov.py      # Confirm chaotic behavior
   ```

## Notes

- The analysis tools use RAM-to-RAM encryption (STDIN/STDOUT mode) for efficiency
- Scalability benchmarks may be limited by available GPU memory
- For very large images (>100 MP), some metrics use sampling to maintain performance
- All tools support both grayscale and color images
