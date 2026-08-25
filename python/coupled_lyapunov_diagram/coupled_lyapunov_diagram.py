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
    from coupled_map.coupled_map import coupled_step, cosine_cosine_map, evolve_ca_16bit
except ImportError:
    from coupled_map import coupled_step, cosine_cosine_map, evolve_ca_16bit

try:
    from coupled_lyapunov.coupled_lyapunov import compute_max_le, d_cosine_cosine
except ImportError:
    from coupled_lyapunov import compute_max_le, d_cosine_cosine


def compute_diag_le(n, rule, r, eps, iterations=500, transition=100):
    """
    Modified LE calculation that incorporates an explicit epsilon factor for the diagram.
    When eps=1, it's the standard coupled model. When eps=0, it's uncoupled.
    """
    xs = np.random.rand(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)
    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    le_sum = 0.0

    # Transient phase
    for _ in range(transition):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    for _ in range(iterations):
        mapped_xs = cosine_cosine_map(xs, r)
        mapped_xs_prime = d_cosine_cosine(xs, r)

        new_xs = np.zeros(n)
        v_next = np.zeros(n)

        for i in range(n):
            evolved = int(ca_states[i])
            v1 = ((evolved >> 8) & 0xFF) / 255.0
            v2 = (evolved & 0xFF) / 255.0

            c_w0, r_w0, l_w0 = v1, (1.0 - v1) * v2, (1.0 - v1) * (1.0 - v2)

            idx_prev = (i - 1) % n
            idx_next = (i + 1) % n

            c_total = (1.0 - eps) + eps * c_w0
            r_total = eps * r_w0
            l_total = eps * l_w0

            new_xs[i] = (c_total * mapped_xs[i]) + \
                        (r_total * mapped_xs[idx_next]) + \
                        (l_total * mapped_xs[idx_prev])

            v_next[i] = (c_total * mapped_xs_prime[i] * v[i]) + \
                        (r_total * mapped_xs_prime[idx_next] * v[idx_next]) + \
                        (l_total * mapped_xs_prime[idx_prev] * v[idx_prev])

        norm = np.linalg.norm(v_next)
        if norm < 1e-20:
            norm = 1e-20
        le_sum += np.log(norm)
        v = v_next / norm

        xs = new_xs
        ca_states = np.array([evolve_ca_16bit(int(s), rule) for s in ca_states], dtype=np.uint16)

    return le_sum / iterations


def main():
    parser = argparse.ArgumentParser(description="Diagrama 2D de Lyapunov para Mapas Acoplados (r vs Acoplamiento eps)")
    parser.add_argument("--n", type=int, default=8, help="Número de mapas caóticos (default: 8)")
    parser.add_argument("--rule", type=int, default=30, help="Regla del CA (default: 30)")
    parser.add_argument("--r_min", type=float, default=2.0, help="Min r (default: 2.0)")
    parser.add_argument("--r_max", type=float, default=6.5, help="Max r (default: 6.5)")
    parser.add_argument("--r_num", type=int, default=30, help="Resolución en rejilla r (default: 30)")
    parser.add_argument("--eps_min", type=float, default=0.0, help="Min acoplamiento eps (default: 0.0)")
    parser.add_argument("--eps_max", type=float, default=1.0, help="Max acoplamiento eps (default: 1.0)")
    parser.add_argument("--eps_num", type=int, default=30, help="Resolución en rejilla eps (default: 30)")
    parser.add_argument("--save", type=str, default="", help="Ruta para guardar la imagen (opcional)")

    args = parser.parse_args()

    rs = np.linspace(args.r_min, args.r_max, args.r_num)
    epsilons = np.linspace(args.eps_min, args.eps_max, args.eps_num)

    matrix = np.zeros((args.eps_num, args.r_num))

    print(f"Generando Diagrama 2D de Lyapunov (r: [{args.r_min}, {args.r_max}], eps: [{args.eps_min}, {args.eps_max}])...")
    for j, r in enumerate(rs):
        for i, eps in enumerate(epsilons):
            matrix[i, j] = compute_diag_le(args.n, args.rule, r, eps, 500, 100)
        if (j + 1) % 5 == 0 or (j + 1) == args.r_num:
            print(f"Progreso: {100*(j+1)/args.r_num:.1f}%")

    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, extent=[args.r_min, args.r_max, args.eps_min, args.eps_max],
                    origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(im, label='Exponente Máximo de Lyapunov')
    plt.title(f"Diagrama 2D de Lyapunov (r vs Acoplamiento eps, n={args.n}, Regla={args.rule})")
    plt.xlabel("Parámetro r")
    plt.ylabel("Fuerza de Acoplamiento (epsilon)")

    plt.axhline(1.0, color='cyan', linestyle='--', alpha=0.7, label='Modelo Acoplado Completo (eps=1.0)')
    plt.legend()

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Diagrama guardado en: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
