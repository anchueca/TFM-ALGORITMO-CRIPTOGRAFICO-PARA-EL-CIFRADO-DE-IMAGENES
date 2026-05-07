#!/usr/bin/env python3
"""
nist_tests.py — Complete NIST SP 800-22 Randomness Tests for Three Sources
============================================================================

Runs the full NIST SP 800-22 statistical test suite (15 tests) on binary data
generated from three different sources:

  1. CML (Coupled Map Lattice)  — Chaotic sequence from the coupled system
  2. Cipher Scheme              — Raw bytes from the encrypted image output
  3. Cellular Automaton          — Isolated 16-bit elementary CA evolution

Tests implemented (complete NIST SP 800-22 suite):
  01. Frequency (Monobit)              09. Maurer's Universal Statistical
  02. Runs                             10. Linear Complexity
  03. Block Frequency                  11. Serial (∇1, ∇2)
  04. Longest Run of Ones              12. Approximate Entropy
  05. Binary Matrix Rank               13. Cumulative Sums
  06. Spectral (DFT)                   14. Random Excursions
  07. Non-overlapping Template         15. Random Excursions Variant
  08. Overlapping Template

Usage:
  # CML + CA only (no cipher executable needed)
  python nist_tests.py

  # All three sources
  python nist_tests.py --exe cuda/bin/cipher.out --image repositorio/set3/lena3.jpg
"""

import argparse
import struct
import os
import sys
import math
import random
import string
import numpy as np
import scipy.special as special
import scipy.fftpack as fft

# Configure matplotlib backend (needed only if coupled_map imports it)
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from coupled_map import coupled_step, evolve_ca_16bit


# ============================================================================
# NIST SP 800-22 Test Implementations (Complete Suite — 15 Tests)
# ============================================================================

def nist_frequency_test(bits):
    """Test 1: Frequency (Monobit) Test."""
    n = len(bits)
    s_n = np.sum(2 * bits - 1)
    s_obs = abs(s_n) / np.sqrt(n)
    return special.erfc(s_obs / np.sqrt(2))


def nist_runs_test(bits):
    """Test 2: Runs Test."""
    n = len(bits)
    pi = np.mean(bits)
    if abs(pi - 0.5) >= (2 / np.sqrt(n)):
        return 0.0
    v_n = 1 + np.sum(bits[:-1] != bits[1:])
    return special.erfc(abs(v_n - 2 * n * pi * (1 - pi)) /
                        (2 * np.sqrt(2 * n) * pi * (1 - pi)))


def nist_block_frequency_test(bits, m=128):
    """Test 3: Block Frequency Test."""
    n = len(bits)
    n_blocks = n // m
    if n_blocks == 0:
        return 0.0
    pi = [np.mean(bits[i * m:(i + 1) * m]) for i in range(n_blocks)]
    chi_sq = 4 * m * np.sum((np.array(pi) - 0.5) ** 2)
    return special.gammaincc(n_blocks / 2, chi_sq / 2)


def nist_longest_run_test(bits):
    """Test 4: Longest Run of Ones in a Block."""
    n = len(bits)
    if n < 6272:
        return 0.0
    # Select parameters based on sequence length (NIST SP 800-22 Table)
    if n < 750000:
        m, K = 8, 3
        pi_vals = [0.2148, 0.3672, 0.2305, 0.1875]
        boundaries = [1, 2, 3, 4]  # <=1, 2, 3, >=4
    else:
        m, K = 128, 5
        pi_vals = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        boundaries = [4, 5, 6, 7, 8, 9]  # <=4, 5, 6, 7, 8, >=9
    n_blocks = n // m
    freq = np.zeros(K + 1)
    for i in range(n_blocks):
        block = bits[i * m:(i + 1) * m]
        max_run = 0
        current_run = 0
        for b in block:
            if b == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        if max_run <= boundaries[0]:
            freq[0] += 1
        elif max_run >= boundaries[-1]:
            freq[K] += 1
        else:
            for j in range(1, K):
                if max_run == boundaries[j]:
                    freq[j] += 1
                    break
    chi_sq = sum((freq[i] - n_blocks * pi_vals[i]) ** 2 /
                 (n_blocks * pi_vals[i]) for i in range(K + 1))
    return special.gammaincc(K / 2.0, chi_sq / 2.0)


def _gf2_rank(matrix):
    """Computes rank of a binary matrix over GF(2)."""
    m = matrix.copy()
    rows, cols = m.shape
    rank = 0
    for col in range(min(rows, cols)):
        # Find pivot
        pivot = None
        for row in range(rank, rows):
            if m[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap
        m[[rank, pivot]] = m[[pivot, rank]]
        # Eliminate
        for row in range(rows):
            if row != rank and m[row, col] == 1:
                m[row] = (m[row] + m[rank]) % 2
        rank += 1
    return rank


def nist_binary_matrix_rank_test(bits):
    """Test 5: Binary Matrix Rank Test (32×32 matrices over GF(2))."""
    n = len(bits)
    M, Q = 32, 32
    n_matrices = n // (M * Q)
    if n_matrices == 0:
        return 0.0
    # Theoretical probabilities for full rank, rank-1, and lower
    p_full = 0.2888
    p_minus1 = 0.5776
    p_rest = 0.1336
    counts = [0, 0, 0]  # [full, rank-1, lower]
    for i in range(n_matrices):
        block = bits[i * M * Q:(i + 1) * M * Q].reshape(M, Q).copy()
        r = _gf2_rank(block)
        if r == M:
            counts[0] += 1
        elif r == M - 1:
            counts[1] += 1
        else:
            counts[2] += 1
    expected = [n_matrices * p for p in [p_full, p_minus1, p_rest]]
    chi_sq = sum((counts[i] - expected[i]) ** 2 / expected[i]
                 for i in range(3) if expected[i] > 0)
    return math.exp(-chi_sq / 2.0)


def nist_spectral_test(bits):
    """Test 6: Discrete Fourier Transform (Spectral) Test."""
    n = len(bits)
    if n % 2 != 0:
        bits = bits[:-1]
        n -= 1
    s = 2 * bits - 1
    dft = fft.fft(s)
    m = np.abs(dft[:n // 2])
    threshold = np.sqrt(np.log(1 / 0.05) * n)
    n_obs = np.sum(m < threshold)
    n_exp = 0.95 * n / 2
    d = (n_obs - n_exp) / np.sqrt(n * 0.95 * 0.05 / 4)
    return special.erfc(abs(d) / np.sqrt(2))


def nist_non_overlapping_template_test(bits, m=9):
    """Test 7: Non-overlapping Template Matching Test."""
    n = len(bits)
    template = np.ones(m, dtype=int)  # Template of m ones
    M = max(m * 10, 1032)  # Block length
    N = n // M
    if N == 0:
        return 0.0
    mu = (M - m + 1) / (2 ** m)
    sigma2 = M * (1.0 / (2 ** m) - (2 * m - 1) / (2 ** (2 * m)))
    if sigma2 <= 0:
        return 0.0
    chi_sq = 0.0
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        count = 0
        j = 0
        while j <= len(block) - m:
            if np.array_equal(block[j:j + m], template):
                count += 1
                j += m  # Non-overlapping: skip template length
            else:
                j += 1
        chi_sq += (count - mu) ** 2 / sigma2
    return special.gammaincc(N / 2.0, chi_sq / 2.0)


def nist_overlapping_template_test(bits, m=5):
    """Test 8: Overlapping Template Matching Test."""
    n = len(bits)
    template = np.ones(m, dtype=int)
    M = 1032
    N = n // M
    if N == 0:
        return 0.0
    lam = (M - m + 1) / (2.0 ** m)
    eta = lam / 2.0
    K = 5
    # Precompute probabilities using NIST SP 800-22 formula
    pi = np.zeros(K + 1)
    for i in range(K):
        pi[i] = math.exp(-eta) * (eta ** i) / math.factorial(i)
    pi[K] = max(1.0 - sum(pi[:K]), 1e-10)
    # Ensure no zero probabilities
    pi = np.maximum(pi, 1e-10)
    freq = np.zeros(K + 1)
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        count = 0
        for j in range(M - m + 1):
            if np.array_equal(block[j:j + m], template):
                count += 1
        idx = min(count, K)
        freq[idx] += 1
    chi_sq = sum((freq[i] - N * pi[i]) ** 2 / (N * pi[i])
                 for i in range(K + 1))
    return special.gammaincc(K / 2.0, chi_sq / 2.0)


def nist_maurers_universal_test(bits):
    """Test 9: Maurer's Universal Statistical Test."""
    n = len(bits)
    if n < 387840:
        L, Q_param = 6, 640
    elif n < 904960:
        L, Q_param = 7, 1280
    else:
        L, Q_param = 8, 2560
    K = n // L - Q_param
    if K <= 0:
        return 0.0
    # Expected values and variance (from NIST SP 800-22 Table)
    expected = {6: 5.2177052, 7: 6.1962507, 8: 7.1836656}
    variance = {6: 2.954, 7: 3.125, 8: 3.238}
    mu = expected.get(L, 7.18)
    c = 0.7 - 0.8 / L + (4 + 32.0 / L) * (K ** (-3.0 / L)) / 15.0
    sigma = c * math.sqrt(variance.get(L, 3.24) / K)
    # Initialize table
    table = np.zeros(2 ** L, dtype=int)
    for i in range(Q_param):
        val = 0
        for j in range(L):
            val = (val << 1) | int(bits[i * L + j])
        table[val] = i + 1
    # Test phase
    fn_sum = 0.0
    for i in range(Q_param, Q_param + K):
        val = 0
        for j in range(L):
            val = (val << 1) | int(bits[i * L + j])
        fn_sum += math.log2(i + 1 - table[val]) if table[val] > 0 else math.log2(i + 1)
        table[val] = i + 1
    fn = fn_sum / K
    return special.erfc(abs(fn - mu) / (math.sqrt(2) * sigma))


def _berlekamp_massey(bits):
    """Berlekamp-Massey algorithm for LFSR complexity over GF(2)."""
    n = len(bits)
    c = np.zeros(n, dtype=int)
    b = np.zeros(n, dtype=int)
    c[0] = 1
    b[0] = 1
    L, m, d_old = 0, -1, 1
    for i in range(n):
        d = bits[i]
        for j in range(1, L + 1):
            d ^= c[j] & bits[i - j]
        if d == 1:
            t = c.copy()
            shift = i - m
            for j in range(n - shift):
                c[j + shift] ^= b[j]
            if L <= i // 2:
                L = i + 1 - L
                m = i
                b = t.copy()
                d_old = d
    return L


def nist_linear_complexity_test(bits, M=500):
    """Test 10: Linear Complexity Test."""
    n = len(bits)
    N = n // M
    if N == 0:
        return 0.0
    K = 6
    pi_vals = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    mu = M / 2.0 + (9 + (-1) ** (M + 1)) / 36.0 - (M / 3.0 + 2.0 / 9.0) / (2 ** M)
    freq = np.zeros(K + 1)
    for i in range(N):
        block = bits[i * M:(i + 1) * M].astype(int)
        Li = _berlekamp_massey(block)
        Ti = (-1) ** M * (Li - mu) + 2.0 / 9.0
        if Ti <= -2.5:
            freq[0] += 1
        elif Ti <= -1.5:
            freq[1] += 1
        elif Ti <= -0.5:
            freq[2] += 1
        elif Ti <= 0.5:
            freq[3] += 1
        elif Ti <= 1.5:
            freq[4] += 1
        elif Ti <= 2.5:
            freq[5] += 1
        else:
            freq[6] += 1
    chi_sq = sum((freq[i] - N * pi_vals[i]) ** 2 / (N * pi_vals[i])
                 for i in range(K + 1) if N * pi_vals[i] > 0)
    return special.gammaincc(K / 2.0, chi_sq / 2.0)


def _psi_sq(bits, m):
    """Helper for Serial and Approximate Entropy tests."""
    n = len(bits)
    if m == 0:
        return 0.0
    # Augment sequence circularly
    augmented = np.concatenate([bits, bits[:m - 1]])
    counts = {}
    for i in range(n):
        pattern = tuple(augmented[i:i + m])
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(v ** 2 for v in counts.values())
    return (2 ** m / n) * total - n


def nist_serial_test(bits, m=16):
    """Test 11: Serial Test. Returns (p_value1, p_value2)."""
    n = len(bits)
    if m >= int(math.log2(n)) - 2:
        m = max(2, int(math.log2(n)) - 3)
    psi_m = _psi_sq(bits, m)
    psi_m1 = _psi_sq(bits, m - 1)
    psi_m2 = _psi_sq(bits, m - 2) if m >= 2 else 0.0
    delta1 = psi_m - psi_m1
    delta2 = psi_m - 2 * psi_m1 + psi_m2
    p1 = special.gammaincc(2 ** (m - 2), delta1 / 2.0) if delta1 > 0 else 1.0
    p2 = special.gammaincc(2 ** (m - 3), delta2 / 2.0) if m >= 3 and delta2 > 0 else 1.0
    return p1, p2


def nist_approximate_entropy_test(bits, m=10):
    """Test 12: Approximate Entropy Test."""
    n = len(bits)
    if m >= int(math.log2(n)) - 5:
        m = max(2, int(math.log2(n)) - 6)

    def phi(block_len):
        if block_len == 0:
            return 0.0
        augmented = np.concatenate([bits, bits[:block_len - 1]])
        counts = {}
        for i in range(n):
            pattern = tuple(augmented[i:i + block_len])
            counts[pattern] = counts.get(pattern, 0) + 1
        total = sum((v / n) * math.log(v / n) for v in counts.values())
        return total

    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    apen = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - apen)
    return special.gammaincc(2 ** (m - 1), chi_sq / 2.0)


def nist_cumulative_sums_test(bits):
    """Test 13: Cumulative Sums (Cusum) Test. Returns min(p_fwd, p_rev)."""
    n = len(bits)
    s = 2 * bits.astype(np.float64) - 1
    # Forward
    cs_fwd = np.cumsum(s)
    z_fwd = np.max(np.abs(cs_fwd))
    # Backward
    cs_rev = np.cumsum(s[::-1])
    z_rev = np.max(np.abs(cs_rev))

    def cusum_p(z):
        sum_val = 0.0
        start = int((-n / z + 1) / 4)
        end = int((n / z - 1) / 4) + 1
        for k in range(start, end + 1):
            sum_val += (special.ndtr((4 * k + 1) * z / np.sqrt(n)) -
                        special.ndtr((4 * k - 1) * z / np.sqrt(n)))
        sum_val2 = 0.0
        start2 = int((-n / z - 3) / 4)
        end2 = int((n / z - 1) / 4) + 1
        for k in range(start2, end2 + 1):
            sum_val2 += (special.ndtr((4 * k + 3) * z / np.sqrt(n)) -
                         special.ndtr((4 * k + 1) * z / np.sqrt(n)))
        return 1.0 - sum_val + sum_val2

    p_fwd = cusum_p(z_fwd)
    p_rev = cusum_p(z_rev)
    return min(p_fwd, p_rev)


def nist_random_excursions_test(bits):
    """Test 14: Random Excursions Test. Returns min p-value across 8 states."""
    n = len(bits)
    s = 2 * bits.astype(int) - 1
    cumsum = np.concatenate([[0], np.cumsum(s)])
    # Find cycles (zero crossings)
    zeros = np.where(cumsum == 0)[0]
    J = len(zeros) - 1
    if J < 500:
        return 0.0  # Not enough cycles
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    # Theoretical probabilities (from NIST SP 800-22 Table)
    pi_table = {
        1: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0312],
        2: [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
        3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804],
        4: [0.8750, 0.0156, 0.0137, 0.0120, 0.0105, 0.0733],
    }
    p_values = []
    for x in states:
        ax = abs(x)
        pi = pi_table[ax]
        freq = np.zeros(6)
        for c in range(J):
            cycle = cumsum[zeros[c]:zeros[c + 1] + 1]
            count = np.sum(cycle == x)
            idx = min(int(count), 5)
            freq[idx] += 1
        chi_sq = sum((freq[k] - J * pi[k]) ** 2 / (J * pi[k])
                     for k in range(6) if J * pi[k] > 0)
        p = special.gammaincc(5 / 2.0, chi_sq / 2.0)
        p_values.append(p)
    return min(p_values) if p_values else 0.0


def nist_random_excursions_variant_test(bits):
    """Test 15: Random Excursions Variant Test. Returns min p-value across 18 states."""
    n = len(bits)
    s = 2 * bits.astype(int) - 1
    cumsum = np.concatenate([[0], np.cumsum(s)])
    zeros = np.where(cumsum == 0)[0]
    J = len(zeros) - 1
    if J < 500:
        return 0.0
    states = list(range(-9, 0)) + list(range(1, 10))
    p_values = []
    for x in states:
        count = np.sum(cumsum == x)
        p = special.erfc(abs(count - J) / np.sqrt(2 * J * (4 * abs(x) - 2)))
        p_values.append(p)
    return min(p_values) if p_values else 0.0


def run_all_tests(bits):
    """Runs all 15 NIST SP 800-22 tests. Returns dict of p-values."""
    results = {
        '01. Frequency (Monobit)': nist_frequency_test(bits),
        '02. Runs': nist_runs_test(bits),
        '03. Block Frequency': nist_block_frequency_test(bits),
        '04. Longest Run': nist_longest_run_test(bits),
        '05. Binary Matrix Rank': nist_binary_matrix_rank_test(bits),
        '06. Spectral (DFT)': nist_spectral_test(bits),
        '07. Non-overlapping Template': nist_non_overlapping_template_test(bits),
        '08. Overlapping Template': nist_overlapping_template_test(bits),
        '09. Maurer Universal': nist_maurers_universal_test(bits),
        '10. Linear Complexity': nist_linear_complexity_test(bits),
        '12. Approx. Entropy': nist_approximate_entropy_test(bits),
        '13. Cumulative Sums': nist_cumulative_sums_test(bits),
        '14. Random Excursions': nist_random_excursions_test(bits),
        '15. Random Excursions Var.': nist_random_excursions_variant_test(bits),
    }
    # Serial test returns two p-values
    p1, p2 = nist_serial_test(bits)
    results['11. Serial (∇1)'] = p1
    results['11. Serial (∇2)'] = p2
    # Sort by key
    return dict(sorted(results.items()))



# ============================================================================
# Binarization / Whitening
# ============================================================================

def binarize_float(val):
    """
    Converts a float to 8 bits using mantissa extraction + XOR whitening.
    """
    float_bits = struct.unpack('>Q', struct.pack('>d', val))[0]
    mantissa = float_bits & ((1 << 52) - 1)
    top32 = mantissa >> (52 - 32)
    b0 = (top32 >> 24) & 0xFF
    b1 = (top32 >> 16) & 0xFF
    b2 = (top32 >> 8) & 0xFF
    b3 = top32 & 0xFF
    result_byte = b0 ^ b1 ^ b2 ^ b3
    return np.array([(result_byte >> i) & 1 for i in range(8)], dtype=np.int32)


# ============================================================================
# Data Generators
# ============================================================================

def generate_cml_bits(n, rule, r, iterations, transition):
    """
    Generates a bit sequence from CML evolution.

    Evolves the coupled map lattice for `transition + iterations` steps,
    then binarizes each of the n chaotic variables at each step.

    Returns:
        bits: numpy array of 0/1 integers
        total_bits: number of bits generated
    """
    xs = np.random.rand(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)

    # Transient
    for _ in range(transition):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    # Collection
    all_bits = []
    for _ in range(iterations):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)
        for x in xs:
            all_bits.extend(binarize_float(x))

    return np.array(all_bits, dtype=np.int32)


def generate_coupled_ca_bits(n, rule, r, iterations, transition):
    """
    Generates a bit sequence from the CA states within the Coupled Map Lattice.
    """
    xs = np.random.rand(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)

    # Transient
    for _ in range(transition):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    # Collection
    all_bits = []
    for _ in range(iterations):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)
        for state in ca_states:
            # Extract only the lower 8 bits
            for bit_pos in range(8):
                all_bits.append((int(state) >> bit_pos) & 1)

    return np.array(all_bits, dtype=np.int32)


def generate_cipher_bits(exe_path, image_path, password, rounds,
                         block_size, automata_steps, transition):
    """
    Generates a bit sequence from the encrypted image output.

    Encrypts the image using the CUDA cipher executable and extracts
    all bytes of the ciphertext as bits.

    Returns:
        bits: numpy array of 0/1 integers, or None on failure
    """
    try:
        import cv2
        import subprocess
    except ImportError as e:
        print(f"  [!] Cannot run cipher test: {e}")
        return None

    if not os.path.exists(exe_path):
        print(f"  [!] Executable not found: {exe_path}")
        return None
    if not os.path.exists(image_path):
        print(f"  [!] Image not found: {image_path}")
        return None

    # Load image
    original_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original_img is None:
        print(f"  [!] Could not read image: {image_path}")
        return None

    # Calculate required key length based on padded dimensions
    rows, base_cols = original_img.shape[:2]
    channels = original_img.shape[2] if len(original_img.shape) > 2 else 1
    unstacked_cols = base_cols * channels if channels == 3 else base_cols
    total_pixels_original = unstacked_cols * rows
    bytes_needed = 5
    pixels_for_meta = math.ceil(bytes_needed / 1.0)
    min_S = math.ceil(math.sqrt(total_pixels_original + pixels_for_meta))
    S = ((min_S + block_size - 1) // block_size) * block_size

    padded_cols = S
    MAX_THREADS = 64
    num_blocks = (padded_cols + MAX_THREADS - 2) // MAX_THREADS - 1
    if num_blocks < 1:
        num_blocks = 1
    total_bytes = ((padded_cols * 2) + 4 +
                   (padded_cols + num_blocks) * 4 +
                   (padded_cols + num_blocks) * 4 + 8)
    required_bits = total_bytes * 8

    # Generate binary password of correct length if not provided
    if password is None:
        password = ''.join(random.choices('01', k=required_bits))
        is_binary = True
    else:
        is_binary = (all(c in '01' for c in password) and len(password) > 100)

    # Encode image
    success, encoded_buffer = cv2.imencode(".tif", original_img)
    if not success:
        print("  [!] Failed to encode image")
        return None

    binary_flag = '1' if is_binary else '0'
    cmd = [
        exe_path, "STDIN", "STDOUT",
        password, str(rounds), '1',  # mode_enc=True
        str(block_size), str(automata_steps), str(transition), "0",
        binary_flag
    ]

    try:
        res = subprocess.run(cmd, input=encoded_buffer.tobytes(),
                             capture_output=True, check=True)
        if not res.stdout:
            print("  [!] Cipher returned empty output")
            return None

        nparr = np.frombuffer(res.stdout, np.uint8)
        ciph_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if ciph_img is None:
            print("  [!] Could not decode cipher output")
            return None

        # Extract bits from ciphertext
        return np.unpackbits(ciph_img.flatten()).astype(np.int32)

    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode('utf-8', errors='ignore')[:200]
        print(f"  [!] Cipher error: {stderr_msg}")
        return None
    except Exception as e:
        print(f"  [!] Unexpected error: {e}")
        return None


# ============================================================================
# Report Formatting
# ============================================================================

def format_result(p_value, threshold=0.01):
    """Formats a p-value with PASS/FAIL indicator."""
    status = "PASS" if p_value > threshold else "FAIL"
    return f"{p_value:.6f}", status


def print_report(results_dict, save_path=None):
    """
    Prints a formatted comparison table of NIST test results.

    Args:
        results_dict: dict of {source_name: {test_name: p_value}}
        save_path: optional path to save the table as text
    """
    sources = list(results_dict.keys())
    test_names = list(next(iter(results_dict.values())).keys())

    # Build table data
    headers = ["NIST Test (SP 800-22)"] + sources + ["Threshold"]
    rows = []

    for test in test_names:
        row = [test]
        for source in sources:
            p = results_dict[source][test]
            if p is None:
                row.append("N/A")
            else:
                val, status = format_result(p)
                row.append(f"{val} [{status}]")
        row.append("> 0.01")
        rows.append(row)

    # Summary row: count passes per source
    summary_row = ["TOTAL PASS"]
    for source in sources:
        passes = sum(1 for test in test_names
                     if results_dict[source][test] is not None
                     and results_dict[source][test] > 0.01)
        total = sum(1 for test in test_names
                    if results_dict[source][test] is not None)
        summary_row.append(f"{passes}/{total}")
    summary_row.append("")
    rows.append(summary_row)

    # Print
    print("\n" + "=" * 90)
    print("  NIST SP 800-22 RANDOMNESS TEST SUITE — COMPARATIVE RESULTS")
    print("=" * 90)

    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    else:
        # Fallback without tabulate
        print(f"{'Test':<25}", end="")
        for s in sources:
            print(f" {s:<22}", end="")
        print()
        print("-" * (25 + 22 * len(sources)))
        for row in rows:
            for cell in row:
                print(f" {str(cell):<22}", end="")
            print()

    print("=" * 90)

    # Metadata
    for source in sources:
        total_tests = sum(1 for t in test_names if results_dict[source][t] is not None)
        passes = sum(1 for t in test_names
                     if results_dict[source][t] is not None
                     and results_dict[source][t] > 0.01)
        if total_tests > 0:
            rate = passes / total_tests * 100
            print(f"  {source}: {passes}/{total_tests} tests passed ({rate:.1f}%)")

    print()

    # Save to file
    if save_path and tabulate:
        with open(save_path, "w") as f:
            f.write("NIST SP 800-22 RANDOMNESS TEST SUITE — COMPARATIVE RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(tabulate(rows, headers=headers, tablefmt="grid"))
            f.write("\n")
        print(f"[+] Report saved to: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NIST SP 800-22 Randomness Tests for CML, Cipher Scheme, and Cellular Automaton"
    )

    # CML parameters
    parser.add_argument("--n", type=int, default=16,
                        help="Number of coupled maps (CML lattice size, default: 16)")
    parser.add_argument("--rule", type=int, default=30,
                        help="Cellular automaton rule (default: 30)")
    parser.add_argument("--r", type=float, default=6.1,
                        help="Chaotic parameter r for CML (default: 6.1)")
    parser.add_argument("--iterations", type=int, default=10000,
                        help="Number of iterations for data generation (default: 10000)")
    parser.add_argument("--transition", type=int, default=200,
                        help="Transient iterations to discard (default: 200)")

    # CA parameters
    parser.add_argument("--ca_initial", type=int, default=0xACE1,
                        help="Initial 16-bit state for the isolated CA (default: 0xACE1)")
    parser.add_argument("--ca_iterations", type=int, default=100000,
                        help="Number of CA iterations (default: 100000)")

    # Cipher parameters
    parser.add_argument("--exe", type=str, default="",
                        help="Path to cipher executable (e.g. cuda/bin/cipher.out)")
    parser.add_argument("--image", type=str, default="",
                        help="Path to input image for cipher test")
    parser.add_argument("--password", type=str, default=None,
                        help="Password for cipher (auto-generated if omitted)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Cipher encryption rounds (default: 3)")
    parser.add_argument("--block-size", type=int, default=8,
                        help="Cipher block size (default: 8)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Automata steps in cipher (default: 20)")
    parser.add_argument("--trans", type=int, default=20,
                        help="Transition length in cipher (default: 20)")

    # Output
    parser.add_argument("--save-binary", action="store_true",
                        help="Save raw binary data to .bin files for use with official NIST sts")
    parser.add_argument("--save", type=str, default="",
                        help="Path to save the report table as .txt")

    args = parser.parse_args()

    all_results = {}

    # --- 1. CML Test ---
    print(f"\n[1/3] Generating CML data (n={args.n}, Rule={args.rule}, "
          f"r={args.r}, {args.iterations} iterations)...")
    cml_bits = generate_cml_bits(args.n, args.rule, args.r,
                                 args.iterations, args.transition)
    print(f"      → {len(cml_bits)} bits generated")
    cml_results = run_all_tests(cml_bits)
    all_results['CML'] = cml_results

    if args.save_binary:
        cml_bytes = np.packbits(cml_bits.astype(np.uint8))
        with open("nist_cml.bin", "wb") as f:
            f.write(cml_bytes.tobytes())
        print(f"      → Binary data saved to nist_cml.bin ({len(cml_bytes)} bytes)")

    # --- 2. Coupled Cellular Automaton Test ---
    print(f"\n[2/3] Generating Coupled CA data (n={args.n}, Rule={args.rule}, "
          f"r={args.r}, {args.ca_iterations} iterations)...")
    ca_bits = generate_coupled_ca_bits(args.n, args.rule, args.r,
                                       args.ca_iterations, args.transition)
    print(f"      → {len(ca_bits)} bits generated")
    ca_results = run_all_tests(ca_bits)
    all_results['Coupled CA'] = ca_results

    if args.save_binary:
        ca_bytes = np.packbits(ca_bits.astype(np.uint8))
        with open("nist_coupled_ca.bin", "wb") as f:
            f.write(ca_bytes.tobytes())
        print(f"      → Binary data saved to nist_coupled_ca.bin ({len(ca_bytes)} bytes)")

    # --- 3. Cipher Scheme Test ---
    if args.exe and args.image:
        print(f"\n[3/3] Generating Cipher data (exe={args.exe}, image={args.image})...")
        cipher_bits = generate_cipher_bits(
            args.exe, args.image, args.password,
            args.rounds, getattr(args, 'block_size', 8),
            args.steps, args.trans
        )
        if cipher_bits is not None:
            print(f"      → {len(cipher_bits)} bits generated")
            cipher_results = run_all_tests(cipher_bits)
            all_results['Cipher Scheme'] = cipher_results

            if args.save_binary:
                cipher_bytes = np.packbits(cipher_bits.astype(np.uint8))
                with open("nist_cipher.bin", "wb") as f:
                    f.write(cipher_bytes.tobytes())
                print(f"      → Binary data saved to nist_cipher.bin ({len(cipher_bytes)} bytes)")
        else:
            print("      → Cipher test skipped (error during encryption)")
    else:
        print(f"\n[3/3] Cipher test skipped (provide --exe and --image to enable)")

    # --- Report ---
    print_report(all_results, save_path=args.save if args.save else None)


if __name__ == "__main__":
    main()
