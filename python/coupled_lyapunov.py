import argparse
import numpy as np
import matplotlib
from coupled_map import coupled_step, cosine_cosine_map

# Configure matplotlib backend for headless environments
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

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
    # Start with a small perturbation vector v
    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    
    le_sum = 0.0
    
    for _ in range(iterations):
        # 1. Update the chaotic state
        # We need the weights for the Jacobian
        # First, evolve CA to get current weights
        new_ca_states = np.array([int((xs[i] * 0xFFFF) % 0xFFFF) for i in range(n)], dtype=np.uint16) # Placeholder for CA update if needed or use internal
        # Actually, let's just use the coupled_step logic but extract weights
        
        # We'll re-calculate the Jacobian J
        # J_ij = dw_i/dx_j
        # For simplicity, we assume weights w_i are fixed for one step of J*v
        
        mapped_xs_prime = d_cosine_cosine(xs, r)
        
        # Calculate weights from current CA states
        weights = []
        for i in range(n):
            evolved = int(ca_states[i])
            v1 = ((evolved >> 8) & 0xFF) / 255.0
            v2 = (evolved & 0xFF) / 255.0
            weights.append((v1, (1.0 - v1) * v2, (1.0 - v1) * (1.0 - v2))) # (c, r, l)
            
        # Linear tangent map (Jacobian multiplication): v_next = J * v
        v_next = np.zeros(n)
        for i in range(n):
            c_w, r_w, l_w = weights[i]
            idx_prev = (i - 1) % n
            idx_next = (i + 1) % n
            
            # x_i(t+1) = c_w * f(x_i(t)) + r_w * f(x_next(t)) + l_w * f(x_prev(t))
            # d x_i(t+1) / d x_j(t) = c_w * f'(x_i(t)) * delta_ij + ...
            v_next[i] = (c_w * mapped_xs_prime[i] * v[i]) + \
                        (r_w * mapped_xs_prime[idx_next] * v[idx_next]) + \
                        (l_w * mapped_xs_prime[idx_prev] * v[idx_prev])
        
        # Normalize v and accumulate expansion rate
        norm = np.linalg.norm(v_next)
        if norm < 1e-20: norm = 1e-20 # Avoid log(0)
        le_sum += np.log(norm)
        v = v_next / norm
        
        # Update the system state
        xs, ca_states = coupled_step(xs, ca_states, r, rule)
        
    return le_sum / iterations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lyapunov Exponent calculation for coupled maps")
    parser.add_argument("--n", type=int, default=16, help="Number of chaotic maps")
    parser.add_argument("--rule", type=int, default=30, help="Rule for CA evolution")
    parser.add_argument("--r_min", type=float, default=2.0, help="Min r")
    parser.add_argument("--r_max", type=float, default=6.5, help="Max r")
    parser.add_argument("--r_num", type=int, default=100, help="Number of r points")
    parser.add_argument("--iterations", type=int, default=500, help="Iterations for LE average")
    parser.add_argument("--transition", type=int, default=100, help="Transition iterations")
    parser.add_argument("--save", type=str, default="", help="Save path")
    
    args = parser.parse_args()
    
    r_values = np.linspace(args.r_min, args.r_max, args.r_num)
    le_values = []
    
    print(f"Calculating LYAPUNOV SPECTRUM (Max LE) for n={args.n}, Rule={args.rule}...")
    for r in r_values:
        le = compute_max_le(args.n, args.rule, r, args.iterations, args.transition)
        le_values.append(le)
        # print(f"r={r:.3f}, LE={le:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, le_values, 'b-', label='Maximum Lyapunov Exponent')
    plt.axhline(0, color='r', linestyle='--', label='Chaos Threshold')
    plt.title(f"Lyapunov Exponent vs Parameter r (n={args.n}, Rule={args.rule})")
    plt.xlabel("Parameter r")
    plt.ylabel("LE")
    plt.legend()
    plt.grid(True)
    
    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.save}")
    else:
        print("Displaying plot (if no window appears, use --save to output a PNG file)...")
        plt.show()
