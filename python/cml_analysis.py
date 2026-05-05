#!/usr/bin/env python3
"""
cml_analysis.py — Unified CML (Coupled Map Lattice) Analysis
=============================================================

Generates a dual-panel figure:
  Left:  Bifurcation diagram of the CML (x_0 values at steady state vs r)
  Right: Maximum Lyapunov Exponent vs r

All core logic is imported from coupled_map.py and coupled_lyapunov.py.
"""

import argparse
import numpy as np
import matplotlib

# Configure matplotlib backend
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from coupled_map import coupled_step
from coupled_lyapunov import compute_max_le


def compute_bifurcation_data(n, rule, r_values, iterations, transition, plot_idx=0):
    """
    Computes bifurcation data for the CML.

    Returns:
        x_coords: list of r values (one per collected point)
        y_coords: list of x_{plot_idx} values at steady state
    """
    x_coords = []
    y_coords = []

    for r in r_values:
        xs = np.random.rand(n)
        ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)

        # Transient phase
        for _ in range(transition):
            xs, ca_states = coupled_step(xs, ca_states, r, rule)

        # Collection phase
        for _ in range(iterations):
            xs, ca_states = coupled_step(xs, ca_states, r, rule)
            x_coords.append(r)
            y_coords.append(xs[plot_idx])

    return x_coords, y_coords


def compute_lyapunov_data(n, rule, r_values, le_iterations, le_transition):
    """
    Computes maximum Lyapunov Exponent for each r value.

    Returns:
        le_values: list of LE values, one per r
    """
    le_values = []
    for r in r_values:
        le = compute_max_le(n, rule, r, le_iterations, le_transition)
        le_values.append(le)
    return le_values


def main():
    parser = argparse.ArgumentParser(
        description="CML Analysis: Bifurcation Diagram + Lyapunov Exponent"
    )
    parser.add_argument("--n", type=int, default=16,
                        help="Number of coupled chaotic maps (default: 16)")
    parser.add_argument("--rule", type=int, default=30,
                        help="Cellular automaton rule (default: 30)")
    parser.add_argument("--r_min", type=float, default=2.0,
                        help="Minimum value of parameter r (default: 2.0)")
    parser.add_argument("--r_max", type=float, default=6.5,
                        help="Maximum value of parameter r (default: 6.5)")
    parser.add_argument("--r_num", type=int, default=800,
                        help="Number of r points (default: 800)")
    parser.add_argument("--bif_iterations", type=int, default=300,
                        help="Iterations to collect per r in bifurcation (default: 300)")
    parser.add_argument("--bif_transition", type=int, default=200,
                        help="Transient iterations for bifurcation (default: 200)")
    parser.add_argument("--le_iterations", type=int, default=800,
                        help="Iterations for Lyapunov Exponent average (default: 800)")
    parser.add_argument("--le_transition", type=int, default=200,
                        help="Transient iterations for Lyapunov calculation (default: 200)")
    parser.add_argument("--plot_idx", type=int, default=0,
                        help="Index of the map to plot in bifurcation (default: 0)")
    parser.add_argument("--save", type=str, default="",
                        help="Path to save the figure (PNG). If empty, displays on screen.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved figure (default: 300)")

    args = parser.parse_args()

    r_values = np.linspace(args.r_min, args.r_max, args.r_num)

    # --- Bifurcation Data ---
    print(f"[1/2] Computing Bifurcation Diagram (n={args.n}, Rule={args.rule}, "
          f"r ∈ [{args.r_min}, {args.r_max}], {args.r_num} points, "
          f"{args.bif_iterations} iters/point)...")
    bif_x, bif_y = compute_bifurcation_data(
        args.n, args.rule, r_values,
        args.bif_iterations, args.bif_transition, args.plot_idx
    )
    print(f"      → {len(bif_y)} bifurcation points collected.")

    # --- Lyapunov Data ---
    print(f"[2/2] Computing Maximum Lyapunov Exponent ({args.le_iterations} iters/point)...")
    le_values = compute_lyapunov_data(
        args.n, args.rule, r_values,
        args.le_iterations, args.le_transition
    )
    print(f"      → {len(le_values)} LE values computed.")

    # --- Plotting ---
    fig, (ax_bif, ax_le) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"CML Analysis — n={args.n}, Rule {args.rule}",
        fontsize=15, fontweight='bold'
    )

    # Left panel: Bifurcation
    ax_bif.scatter(bif_x, bif_y, s=0.2, c='black', alpha=0.4, marker='o', lw=0)
    ax_bif.set_title("Bifurcation Diagram")
    ax_bif.set_xlabel("Parameter r")
    ax_bif.set_ylabel(f"$x_{{{args.plot_idx}}}$")
    ax_bif.set_xlim(args.r_min, args.r_max)
    ax_bif.set_ylim(0, 1)
    ax_bif.grid(True, alpha=0.3)

    # Right panel: Lyapunov Exponent
    ax_le.plot(r_values, le_values, 'b-', linewidth=0.8, label='Max LE')
    ax_le.axhline(0, color='r', linestyle='--', linewidth=1.0, label='Chaos threshold (LE=0)')
    ax_le.fill_between(r_values, le_values, 0,
                       where=[le > 0 for le in le_values],
                       color='red', alpha=0.15, label='Chaotic regime')
    ax_le.fill_between(r_values, le_values, 0,
                       where=[le <= 0 for le in le_values],
                       color='blue', alpha=0.10, label='Stable regime')
    ax_le.set_title("Maximum Lyapunov Exponent")
    ax_le.set_xlabel("Parameter r")
    ax_le.set_ylabel("LE")
    ax_le.set_xlim(args.r_min, args.r_max)
    ax_le.legend(fontsize=8)
    ax_le.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if args.save:
        plt.savefig(args.save, dpi=args.dpi, bbox_inches='tight')
        print(f"\n[+] Figure saved to: {args.save}")
    else:
        print("\nDisplaying plot (if no window appears, use --save to output a PNG)...")
        plt.show()


if __name__ == "__main__":
    main()
