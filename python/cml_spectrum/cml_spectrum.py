#!/usr/bin/env python3
"""
cml_spectrum.py — Full Lyapunov Spectrum, KS Entropy & Kaplan-Yorke Dimension
=============================================================================

Calculates the complete spectrum of N Lyapunov exponents for the Coupled Map Lattice (CML)
using continuous QR decomposition of the system tangent space (Jacobian product).

Computes:
  1. Full spectrum of N exponents: λ_1 >= λ_2 >= ... >= λ_N
  2. Kolmogorov-Sinai (KS) Entropy: h_KS = sum_{λ_i > 0} λ_i
  3. Kaplan-Yorke Dimension (Attractor Dimension): D_KY = j + (sum_{i=1}^j λ_i) / |λ_{j+1}|
"""

import os
import sys
import argparse
import numpy as np
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
    from coupled_map.coupled_map import coupled_step, cosine_cosine_map
except ImportError:
    from coupled_map import coupled_step, cosine_cosine_map

try:
    from coupled_lyapunov.coupled_lyapunov import d_cosine_cosine
except ImportError:
    def d_cosine_cosine(x, r):
        dx = 1e-7
        return (cosine_cosine_map(x + dx, r) - cosine_cosine_map(x - dx, r)) / (2 * dx)


def compute_full_lyapunov_spectrum(n, rule, r, iterations=1000, transition=200):
    """
    Computes all N Lyapunov exponents using QR decomposition.
    Returns array of N exponents sorted in descending order.
    """
    xs = np.random.rand(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)

    # Transient phase
    for _ in range(transition):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    # Initialize Q as identity matrix N x N
    Q = np.eye(n)
    le_sums = np.zeros(n)

    for _ in range(iterations):
        mapped_xs_prime = d_cosine_cosine(xs, r)

        # Build Jacobian matrix J of dimension N x N
        J = np.zeros((n, n))
        for i in range(n):
            evolved = int(ca_states[i])
            v1 = ((evolved >> 8) & 0xFF) / 255.0
            v2 = (evolved & 0xFF) / 255.0
            c_w, r_w, l_w = v1, (1.0 - v1) * v2, (1.0 - v1) * (1.0 - v2)

            idx_prev = (i - 1) % n
            idx_next = (i + 1) % n

            J[i, i] += c_w * mapped_xs_prime[i]
            J[i, idx_next] += r_w * mapped_xs_prime[idx_next]
            J[i, idx_prev] += l_w * mapped_xs_prime[idx_prev]

        # Advance tangent space: Z = J * Q
        Z = J @ Q

        # QR decomposition: Z = Q * R
        Q, R = np.linalg.qr(Z)

        # Accumulate log of diagonal elements of R
        diag_R = np.abs(np.diag(R))
        diag_R = np.maximum(diag_R, 1e-20)
        le_sums += np.log(diag_R)

        # Evolve CML system state
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    spectrum = le_sums / iterations
    spectrum = np.sort(spectrum)[::-1]  # Descending order
    return spectrum


def calculate_ks_entropy(spectrum):
    """Kolmogorov-Sinai Entropy: Sum of positive Lyapunov exponents."""
    pos_les = spectrum[spectrum > 0]
    return np.sum(pos_les)


def calculate_kaplan_yorke_dim(spectrum):
    """
    Kaplan-Yorke (KY) Attractor Dimension:
    D_KY = j + (sum_{i=1}^j λ_i) / |λ_{j+1}|
    """
    n = len(spectrum)
    cum_sum = np.cumsum(spectrum)

    if cum_sum[0] < 0:
        return 0.0

    j = -1
    for i in range(n):
        if cum_sum[i] >= 0:
            j = i
        else:
            break

    if j == n - 1:
        return float(n)

    sum_j = cum_sum[j]
    abs_next = abs(spectrum[j + 1])
    if abs_next < 1e-12:
        return float(j + 1)

    d_ky = (j + 1) + (sum_j / abs_next)
    return d_ky


def main():
    parser = argparse.ArgumentParser(description="Espectro Completo de Lyapunov, Entropía KS y Dimensión de Kaplan-Yorke del CML")
    parser.add_argument("--n", type=int, default=16, help="Número de mapas acoplados N (default: 16)")
    parser.add_argument("--rule", type=int, default=30, help="Regla del autómata celular (default: 30)")
    parser.add_argument("--r", type=float, default=6.1, help="Parámetro r de caos (default: 6.1)")
    parser.add_argument("--iterations", type=int, default=800, help="Iteraciones para promediar (default: 800)")
    parser.add_argument("--transition", type=int, default=200, help="Iteraciones transitorias descartadas (default: 200)")
    parser.add_argument("--save", type=str, default="", help="Ruta de guardado de gráfico PNG (opcional)")

    args = parser.parse_args()

    print(f"Calculando Espectro Completo de Lyapunov (N={args.n}, r={args.r}, Regla={args.rule})...")
    spectrum = compute_full_lyapunov_spectrum(args.n, args.rule, args.r, args.iterations, args.transition)

    ks_entropy = calculate_ks_entropy(spectrum)
    d_ky = calculate_kaplan_yorke_dim(spectrum)

    print("\n" + "=" * 60)
    print(f"ESPECTRO DE LYAPUNOV COMPLETO (N={args.n}):")
    for i, val in enumerate(spectrum):
        sign = "+" if val > 0 else "-"
        print(f"  λ_{i+1:<2d} = {val: .6f}  [{sign}]")

    print("-" * 60)
    print(f"Entropía de Kolmogorov-Sinai (h_KS) : {ks_entropy:.6f}")
    print(f"Dimensión de Kaplan-Yorke (D_KY)     : {d_ky:.6f} / {args.n}")
    print("=" * 60 + "\n")

    # Plot spectrum bar chart + cumulative sum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Espectro Completo de Lyapunov del CML (N={args.n}, r={args.r}, Regla={args.rule})", fontsize=14, fontweight='bold')

    indices = np.arange(1, args.n + 1)
    colors = ['crimson' if val > 0 else 'dodgerblue' for val in spectrum]

    ax1.bar(indices, spectrum, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel("Índice del Exponente (i)")
    ax1.set_ylabel("Exponente de Lyapunov (λ_i)")
    ax1.set_title("Espectro de Exponentes (λ_1 ... λ_N)")
    ax1.set_xticks(indices)
    ax1.grid(True, alpha=0.3)

    cum_sums = np.cumsum(spectrum)
    ax2.plot(indices, cum_sums, 'g-o', linewidth=1.5, markersize=4, label="Suma Acumulada ∑ λ_i")
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, label="Línea Cero")
    ax2.axvline(d_ky, color='purple', linestyle=':', linewidth=2, label=f"D_KY = {d_ky:.2f}")
    ax2.set_xlabel("Número de Exponentes (j)")
    ax2.set_ylabel("Suma Acumulada ∑_{i=1}^j λ_i")
    ax2.set_title("Dimensión de Kaplan-Yorke (D_KY)")
    ax2.set_xticks(indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Gráfico del espectro guardado en: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
