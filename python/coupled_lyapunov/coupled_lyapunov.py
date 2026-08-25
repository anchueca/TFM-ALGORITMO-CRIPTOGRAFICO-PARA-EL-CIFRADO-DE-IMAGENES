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

try:
    from coupled_map.coupled_map import coupled_step, cosine_cosine_map
except ImportError:
    from coupled_map import coupled_step, cosine_cosine_map


def d_cosine_cosine(x, r):
    """
    Numerical derivative of the cosine-cosine map w.r.t x.
    Used for Jacobian calculation in Lyapunov Exponent estimation.
    """
    dx = 1e-7
    diff = (cosine_cosine_map(x + dx, r) - cosine_cosine_map(x - dx, r)) / (2 * dx)
    return diff


def compute_max_le(n, rule, r, iterations=1000, transition=200):
    """
    Computes the maximum Lyapunov Exponent using the Benettin algorithm.
    """
    xs = np.random.rand(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)

    # Transient phase
    for _ in range(transition):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    # Lyapunov estimation
    v = np.random.rand(n)
    v /= np.linalg.norm(v)

    le_sum = 0.0

    for _ in range(iterations):
        mapped_xs_prime = d_cosine_cosine(xs, r)

        # Calculate weights from current CA states
        weights = []
        for i in range(n):
            evolved = int(ca_states[i])
            v1 = ((evolved >> 8) & 0xFF) / 255.0
            v2 = (evolved & 0xFF) / 255.0
            weights.append((v1, (1.0 - v1) * v2, (1.0 - v1) * (1.0 - v2)))

        # Linear tangent map (Jacobian multiplication): v_next = J * v
        v_next = np.zeros(n)
        for i in range(n):
            c_w, r_w, l_w = weights[i]
            idx_prev = (i - 1) % n
            idx_next = (i + 1) % n

            v_next[i] = (c_w * mapped_xs_prime[i] * v[i]) + \
                        (r_w * mapped_xs_prime[idx_next] * v[idx_next]) + \
                        (l_w * mapped_xs_prime[idx_prev] * v[idx_prev])

        # Normalize v and accumulate expansion rate
        norm = np.linalg.norm(v_next)
        if norm < 1e-20:
            norm = 1e-20
        le_sum += np.log(norm)
        v = v_next / norm

        # Update system state
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    return le_sum / iterations


def main():
    parser = argparse.ArgumentParser(description="Cálculo del Espectro de Lyapunov para Mapas Acoplados (CML)")
    parser.add_argument("--n", type=int, default=16, help="Número de mapas caóticos (default: 16)")
    parser.add_argument("--rule", type=int, default=30, help="Regla para la evolución del CA (default: 30)")
    parser.add_argument("--r_min", type=float, default=2.0, help="Valor mínimo de r (default: 2.0)")
    parser.add_argument("--r_max", type=float, default=6.5, help="Valor máximo de r (default: 6.5)")
    parser.add_argument("--r_num", type=int, default=100, help="Número de puntos en el barrido de r (default: 100)")
    parser.add_argument("--iterations", type=int, default=500, help="Iteraciones para promediar LE (default: 500)")
    parser.add_argument("--transition", type=int, default=100, help="Iteraciones transitorias (default: 100)")
    parser.add_argument("--save", type=str, default="", help="Ruta para guardar el gráfico (opcional)")

    args = parser.parse_args()

    r_values = np.linspace(args.r_min, args.r_max, args.r_num)
    le_values = []

    print(f"Calculando Espectro de Lyapunov (Max LE) para n={args.n}, Regla={args.rule}...")
    for r in r_values:
        le = compute_max_le(args.n, args.rule, r, args.iterations, args.transition)
        le_values.append(le)

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, le_values, 'b-', label='Exponente Máximo de Lyapunov')
    plt.axhline(0, color='r', linestyle='--', label='Umbral de Caos (LE=0)')
    plt.title(f"Exponente de Lyapunov vs Parámetro r (n={args.n}, Regla={args.rule})")
    plt.xlabel("Parámetro r")
    plt.ylabel("LE")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
