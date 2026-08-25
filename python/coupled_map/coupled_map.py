#!/usr/bin/env python3
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


def cosine_cosine_map(x, r):
    """
    Chaotic function: |cos(pi * r * cos(pi * t) * t)| where t = r + 3*x^2
    Matches CUDA implementation in kernels.cuh
    """
    t = r + 3.0 * x * x
    return np.abs(np.cos(np.pi * r * np.cos(np.pi * t) * t))


def binarize_float_scalar(val):
    """Equivalent to convertToBitStream in CUDA for Python scalars."""
    import struct
    float_bits = struct.unpack('>Q', struct.pack('>d', float(val)))[0]
    mantissa = float_bits & ((1 << 52) - 1)
    top32 = mantissa >> (52 - 32)
    b0 = (top32 >> 24) & 0xFF
    b1 = (top32 >> 16) & 0xFF
    b2 = (top32 >> 8) & 0xFF
    b3 = top32 & 0xFF
    return b0 ^ b1 ^ b2 ^ b3


def evolve_ca_16bit(state, rule=30):
    """
    Evolves a 16-bit CA state using bitwise operations.
    Matches CUDA evolve_16bit_isolated in automataKernel.cuh
    """
    L = ((state >> 1) | (state << 15)) & 0xFFFF
    R = ((state << 1) | (state >> 15)) & 0xFFFF
    C = state & 0xFFFF

    if rule == 30:
        return (L ^ (C | R)) & 0xFFFF

    next_state = 0
    for p in range(8):
        if (rule >> p) & 1:
            term = 0xFFFF
            term &= L if (p & 4) else ~L
            term &= C if (p & 2) else ~C
            term &= R if (p & 1) else ~R
            next_state |= (term & 0xFFFF)
    return next_state


def coupled_step(xs, ca_states, r, rule=30):
    """
    Performs one step of the coupled map system.
    Returns (new_xs, new_ca_states).
    """
    n = len(xs)
    mapped_xs = cosine_cosine_map(xs, r)

    new_ca_states = np.zeros_like(ca_states, dtype=np.uint16)
    new_xs = np.zeros_like(xs)

    for i in range(n):
        idx_prev = (i - 1) % n
        idx_next = (i + 1) % n

        evolved = evolve_ca_16bit(int(ca_states[i]), rule)

        v1 = ((evolved >> 8) & 0xFF) / 255.0
        v2 = (evolved & 0xFF) / 255.0

        c_influence = v1
        rest = 1.0 - v1
        r_influence = rest * v2
        l_influence = rest * (1.0 - v2)

        new_xs[i] = (mapped_xs[i] * c_influence) + \
                     (mapped_xs[idx_next] * r_influence) + \
                     (mapped_xs[idx_prev] * l_influence)

        noise = binarize_float_scalar(mapped_xs[i])
        noise16 = (noise << 8) | noise
        new_ca_states[i] = (evolved ^ noise16) & 0xFFFF

    return new_xs, new_ca_states


def main():
    parser = argparse.ArgumentParser(description="Simulación interactiva del sistema CML (Coupled Map Lattice)")
    parser.add_argument("--n", type=int, default=8, help="Número de celdas acopladas (default: 8)")
    parser.add_argument("--rule", type=int, default=30, help="Regla del autómata celular (default: 30)")
    parser.add_argument("--r", type=float, default=6.1, help="Parámetro r de caos (default: 6.1)")
    parser.add_argument("--iterations", type=int, default=150, help="Número de iteraciones a simular (default: 150)")
    parser.add_argument("--save", type=str, default="", help="Ruta de guardado de gráfico (opcional)")

    args = parser.parse_args()

    xs = np.random.rand(args.n)
    ca_states = np.random.randint(0, 0xFFFF, args.n, dtype=np.uint16)

    history = np.zeros((args.iterations, args.n))

    for t in range(args.iterations):
        history[t, :] = xs
        xs, ca_states = coupled_step(xs, ca_states, args.r, args.rule)

    plt.figure(figsize=(12, 6))
    for i in range(min(args.n, 8)):
        plt.plot(range(args.iterations), history[:, i], label=f"Celda x_{i}", alpha=0.8, linewidth=1.2)

    plt.title(f"Simulación Temporal de CML (n={args.n}, r={args.r}, Regla={args.rule})", fontsize=14)
    plt.xlabel("Paso de Tiempo (Iteración)")
    plt.ylabel("Estado Caótico x")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Gráfico CML guardado en: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
