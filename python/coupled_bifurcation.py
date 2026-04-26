import argparse
import numpy as np
import matplotlib
from coupled_map import coupled_step

# Configure matplotlib backend for headless environments
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

def bifurcation_coupled(n, rule, r_min, r_max, r_num, iterations, transition, initial_x, initial_ca_state, plot_idx=0):
    r_values = np.linspace(r_min, r_max, r_num)
    
    x_coords = []
    y_coords = []
    
    print(f"Generating Bifurcation Diagram for n={n}, Rule={rule}, r in [{r_min}, {r_max}]...")
    
    for r in r_values:
        # Initialize with full random conditions to show the attractor's breadth
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

    print(f"Plotting {len(y_coords)} points...")
    
    plt.figure(figsize=(14, 10))
    # Use larger markers and higher alpha for better visibility
    plt.scatter(x_coords, y_coords, s=1.0, c='black', alpha=0.5, marker='o', lw=0)
    
    plt.title(f"Bifurcation Diagram (Coupled Map Lattice, n={n}, Rule={rule})")
    plt.xlabel("Parameter r")
    plt.ylabel(f"x_{plot_idx}")
    plt.xlim(r_min, r_max)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bifurcation diagram for coupled maps")
    parser.add_argument("--n", type=int, default=16, help="Number of chaotic maps")
    parser.add_argument("--rule", type=int, default=30, help="Rule for CA evolution")
    parser.add_argument("--r_min", type=float, default=2.0, help="Min r")
    parser.add_argument("--r_max", type=float, default=6.5, help="Max r")
    parser.add_argument("--r_num", type=int, default=800, help="Number of r points")
    parser.add_argument("--iterations", type=int, default=200, help="Iterations per r")
    parser.add_argument("--transition", type=int, default=150, help="Transition iterations")
    parser.add_argument("--initial_x", type=float, default=0.1, help="Initial condition for x")
    parser.add_argument("--initial_ca", type=int, default=0x1234, help="Initial condition for CA")
    parser.add_argument("--plot_idx", type=int, default=0, help="Index of map to plot")
    parser.add_argument("--save", type=str, default="", help="Save path")
    
    args = parser.parse_args()
    
    bifurcation_coupled(args.n, args.rule, args.r_min, args.r_max, args.r_num, 
                       args.iterations, args.transition, args.initial_x, args.initial_ca, args.plot_idx)
    
    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Diagram saved to {args.save}")
    else:
        print("Displaying plot (if no window appears, use --save to output a PNG file)...")
        plt.show()
