#!/usr/bin/env python3
"""
nist_tests.py — Complete NIST SP 800-22 Randomness Tests for Three Sources
============================================================================

Runs the full NIST SP 800-22 statistical test suite (15 tests) on binary data
generated from three different sources:

  1. CML (Coupled Map Lattice)  — Chaotic sequence from the coupled system
  2. Cipher Scheme              — Raw bytes from the encrypted image output
  3. Cellular Automaton          — Isolated 16-bit elementary CA evolution
"""

import os
import sys
import argparse
import struct
import math
import random
import string
import numpy as np
import scipy.special as special
import scipy.fftpack as fft
import matplotlib

try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    from coupled_map.coupled_map import coupled_step, evolve_ca_16bit
except ImportError:
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
    if n < 750000:
        m, K = 8, 3
        pi_vals = [0.2148, 0.3672, 0.2305, 0.1875]
        boundaries = [1, 2, 3, 4]
    else:
        m, K = 128, 5
        pi_vals = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        boundaries = [4, 5, 6, 7, 8, 9]
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
        pivot = None
        for row in range(rank, rows):
            if m[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(rows):
            if row != rank and m[row, col] == 1:
                m[row] = (m[row] + m[rank]) % 2
        rank += 1
    return rank


def nist_binary_matrix_rank_test(bits):
    """Test 5: Binary Matrix Rank Test."""
    n = len(bits)
    M, Q = 32, 32
    n_matrices = n // (M * Q)
    if n_matrices == 0:
        return 0.0
    p_full = 0.2888
    p_minus1 = 0.5776
    p_rest = 0.1336
    counts = [0, 0, 0]
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
    template = np.ones(m, dtype=int)
    M = max(m * 10, 1032)
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
                j += m
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
    pi = np.zeros(K + 1)
    for i in range(K):
        pi[i] = math.exp(-eta) * (eta ** i) / math.factorial(i)
    pi[K] = max(1.0 - sum(pi[:K]), 1e-10)
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
    expected = {6: 5.2177052, 7: 6.1962507, 8: 7.1836656}
    variance = {6: 2.954, 7: 3.125, 8: 3.238}
    mu = expected.get(L, 7.18)
    c = 0.7 - 0.8 / L + (4 + 32.0 / L) * (K ** (-3.0 / L)) / 15.0
    sigma = c * math.sqrt(variance.get(L, 3.24) / K)
    table = np.zeros(2 ** L, dtype=int)
    for i in range(Q_param):
        val = 0
        for j in range(L):
            val = (val << 1) | int(bits[i * L + j])
        table[val] = i + 1
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
    """Berlekamp-Massey algorithm."""
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
    n = len(bits)
    if m == 0:
        return 0.0
    augmented = np.concatenate([bits, bits[:m - 1]])
    counts = {}
    for i in range(n):
        pattern = tuple(augmented[i:i + m])
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(v ** 2 for v in counts.values())
    return (2 ** m / n) * total - n


def nist_serial_test(bits, m=16):
    """Test 11: Serial Test."""
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
    """Test 13: Cumulative Sums Test."""
    n = len(bits)
    s = 2 * bits.astype(np.float64) - 1
    cs_fwd = np.cumsum(s)
    z_fwd = np.max(np.abs(cs_fwd))
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
    """Test 14: Random Excursions Test."""
    n = len(bits)
    s = 2 * bits.astype(int) - 1
    cumsum = np.concatenate([[0], np.cumsum(s)])
    zeros = np.where(cumsum == 0)[0]
    J = len(zeros) - 1
    if J < 500:
        return 0.0
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
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
    """Test 15: Random Excursions Variant Test."""
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
    p1, p2 = nist_serial_test(bits)
    results['11. Serial (∇1)'] = p1
    results['11. Serial (∇2)'] = p2
    return dict(sorted(results.items()))


def binarize_float(val):
    float_bits = struct.unpack('>Q', struct.pack('>d', val))[0]
    mantissa = float_bits & ((1 << 52) - 1)
    top32 = mantissa >> (52 - 32)
    b0 = (top32 >> 24) & 0xFF
    b1 = (top32 >> 16) & 0xFF
    b2 = (top32 >> 8) & 0xFF
    b3 = top32 & 0xFF
    result_byte = b0 ^ b1 ^ b2 ^ b3
    return np.array([(result_byte >> i) & 1 for i in range(8)], dtype=np.int32)


def generate_cml_bits(n, rule, r, iterations, transition):
    xs = np.random.rand(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)

    for _ in range(transition):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    all_bits = []
    for _ in range(iterations):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)
        for x in xs:
            all_bits.extend(binarize_float(x))

    return np.array(all_bits, dtype=np.int32)


def generate_coupled_ca_bits(n, rule, r, iterations, transition):
    xs = np.random.rand(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)

    for _ in range(transition):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    all_bits = []
    for _ in range(iterations):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)
        for state in ca_states:
            for bit_pos in range(8):
                all_bits.append((int(state) >> bit_pos) & 1)

    return np.array(all_bits, dtype=np.int32)


def plot_nist_interactive(results_dict, save_path=None):
    sources = list(results_dict.keys())
    test_names = list(next(iter(results_dict.values())).keys())

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(test_names))
    width = 0.35

    for i, source in enumerate(sources):
        p_vals = [results_dict[source][t] for t in test_names]
        bars = ax.bar(x + (i - 0.5) * width, p_vals, width, label=source, alpha=0.85)
        # Highlight pass/fail
        for bar, p in zip(bars, p_vals):
            if p <= 0.01:
                bar.set_color('red')

    ax.axhline(0.01, color='red', linestyle='--', linewidth=1.5, label='Umbral NIST (α = 0.01)')
    ax.set_ylabel('p-value', fontsize=12)
    ax.set_title('Resultados de la Suite NIST SP 800-22 (Evaluación de Aleatoriedad)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico NIST guardado en: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="NIST SP 800-22 Randomness Tests for CML and Cellular Automaton"
    )
    parser.add_argument("--n", type=int, default=16, help="Tamaño de la red CML (default: 16)")
    parser.add_argument("--rule", type=int, default=30, help="Regla del CA (default: 30)")
    parser.add_argument("--r", type=float, default=6.1, help="Parámetro r de caos (default: 6.1)")
    parser.add_argument("--iterations", type=int, default=5000, help="Iteraciones para generar datos (default: 5000)")
    parser.add_argument("--transition", type=int, default=200, help="Iteraciones transitorias descartadas (default: 200)")
    parser.add_argument("--save", type=str, default="", help="Ruta de guardado (opcional)")

    args = parser.parse_args()

    print(f"Generando secuencias de bits (CML y CA, {args.iterations} iteraciones)...")
    cml_bits = generate_cml_bits(args.n, args.rule, args.r, args.iterations, args.transition)
    ca_bits = generate_coupled_ca_bits(args.n, args.rule, args.r, args.iterations, args.transition)

    print("Ejecutando suite NIST SP 800-22...")
    cml_results = run_all_tests(cml_bits)
    ca_results = run_all_tests(ca_bits)

    results_dict = {
        'CML (Caos)': cml_results,
        'Autómata Celular': ca_results
    }

    # Print text summary
    print("\n--- Resultados NIST SP 800-22 ---")
    for test, p_val in cml_results.items():
        status = "PASS" if p_val > 0.01 else "FAIL"
        print(f"{test:<30}: p-val = {p_val:.6f} [{status}]")

    plot_nist_interactive(results_dict, save_path=args.save)


if __name__ == "__main__":
    main()
