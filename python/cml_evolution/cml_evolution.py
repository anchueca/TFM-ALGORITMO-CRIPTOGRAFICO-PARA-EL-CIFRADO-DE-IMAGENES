#!/usr/bin/env bash
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
    from coupled_map.coupled_map import coupled_step
except ImportError:
    from coupled_map import coupled_step


def plot_evolution(n, rule, r, iterations, transition, initial_x, initial_ca, save=""):
    xs = np.random.rand(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)

    # Transient phase
    for _ in range(transition):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    evolution = np.zeros((iterations, n))

    # Collection phase
    for i in range(iterations):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)
        evolution[i, :] = xs

    plt.figure(figsize=(10, 8))
    im = plt.imshow(evolution, aspect='auto', cmap='magma', origin='lower', extent=[0, n - 1, 0, iterations])
    plt.colorbar(im, label='Valor x [0, 1]')
    plt.xlabel('Índice del Mapa (Espacio)')
    plt.ylabel('Iteración (Tiempo)')
    plt.title(f'Evolución Espacio-Tiempo (Coupled Map Lattice, n={n}, r={r}, Regla={rule})')

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"Gráfico de evolución guardado en: {save}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gráfico de evolución espacio-tiempo para mapas acoplados (CML)")
    parser.add_argument("--n", type=int, default=16, help="Número de mapas caóticos (default: 16)")
    parser.add_argument("--rule", type=int, default=30, help="Regla del autómata celular (default: 30)")
    parser.add_argument("--r", type=float, default=4.5, help="Parámetro r (default: 4.5)")
    parser.add_argument("--iterations", type=int, default=200, help="Iteraciones a graficar (default: 200)")
    parser.add_argument("--transition", type=int, default=100, help="Iteraciones transitorias (default: 100)")
    parser.add_argument("--initial_x", type=float, default=0.1, help="Condición inicial x (default: 0.1)")
    parser.add_argument("--initial_ca", type=int, default=0x1234, help="Condición inicial CA (default: 0x1234)")
    parser.add_argument("--save", type=str, default="", help="Ruta de guardado (opcional)")

    args = parser.parse_args()

    plot_evolution(args.n, args.rule, args.r, args.iterations, args.transition,
                   args.initial_x, args.initial_ca, args.save)
