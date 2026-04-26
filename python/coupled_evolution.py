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
        
    plt.figure(figsize=(12, 10))
    im = plt.imshow(evolution, aspect='auto', cmap='magma', origin='lower', extent=[0, n-1, 0, iterations])
    plt.colorbar(im, label='x value [0, 1]')
    plt.xlabel('Map Index (Space)')
    plt.ylabel('Iteration (Time)')
    plt.title(f'Spacetime Evolution (Coupled Map Lattice, n={n}, r={r}, Rule={rule})')
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"Evolution plot saved to {save}")
    else:
        print("Displaying plot (if no window appears, use --save to output a PNG file)...")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spacetime evolution plot for coupled maps")
    parser.add_argument("--n", type=int, default=16, help="Number of chaotic maps")
    parser.add_argument("--rule", type=int, default=30, help="Rule for CA evolution")
    parser.add_argument("--r", type=float, default=4.5, help="Parameter r")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations to plot")
    parser.add_argument("--transition", type=int, default=100, help="Transition iterations")
    parser.add_argument("--initial_x", type=float, default=0.1, help="Initial condition for x")
    parser.add_argument("--initial_ca", type=int, default=0x1234, help="Initial condition for CA")
    parser.add_argument("--save", type=str, default="", help="Save path")
    
    args = parser.parse_args()
    
    plot_evolution(args.n, args.rule, args.r, args.iterations, args.transition, 
                  args.initial_x, args.initial_ca, args.save)
