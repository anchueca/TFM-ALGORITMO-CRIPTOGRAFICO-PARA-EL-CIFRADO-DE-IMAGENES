import argparse
import numpy as np
import matplotlib
from coupled_map import coupled_step, cosine_cosine_map
from coupled_lyapunov import compute_max_le, d_cosine_cosine

# Configure matplotlib backend for headless environments
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

def compute_diag_le(n, rule, r, eps, iterations=500, transition=100):
    """
    Modified LE calculation that incorporates an explicit epsilon factor for the diagram.
    When eps=1, it's the standard coupled model. When eps=0, it's uncoupled.
    """
    xs = np.random.rand(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)
    v = np.random.rand(n); v /= np.linalg.norm(v)
    le_sum = 0.0
    
    # Pre-calculate transient phase
    for _ in range(transition):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)
        
    for _ in range(iterations):
        mapped_xs = cosine_cosine_map(xs, r)
        mapped_xs_prime = d_cosine_cosine(xs, r)
        
        # New xs and Jacobian calculation with epsilon
        new_xs = np.zeros(n)
        v_next = np.zeros(n)
        
        for i in range(n):
            evolved = int(ca_states[i])
            v1 = ((evolved >> 8) & 0xFF) / 255.0
            v2 = (evolved & 0xFF) / 255.0
            
            c_w0, r_w0, l_w0 = v1, (1.0 - v1)*v2, (1.0 - v1)*(1.0-v2)
            
            # Epsilon modulation: (1-eps)*self + eps*weighted_neighbors
            # x_i' = (1-eps)*f(x_i) + eps*(w_c*f(x_i) + w_r*f(x_r) + w_l*f(x_l))
            
            idx_prev = (i - 1) % n
            idx_next = (i + 1) % n
            
            # Simplified weights
            # c_total = (1-eps) + eps*c_w0
            # r_total = eps*r_w0
            # l_total = eps*l_w0
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
        if norm < 1e-20: norm = 1e-20
        le_sum += np.log(norm)
        v = v_next / norm
        
        xs = new_xs
        # Evolve CA properly for the actual calculation phase
        from coupled_map import evolve_ca_16bit
        ca_states = np.array([evolve_ca_16bit(int(s), rule) for s in ca_states], dtype=np.uint16)
        
    return le_sum / iterations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Lyapunov Diagram for coupled maps")
    parser.add_argument("--n", type=int, default=8, help="Number of chaotic maps (keep small for 2D)")
    parser.add_argument("--rule", type=int, default=30, help="Rule for CA evolution")
    parser.add_argument("--r_min", type=float, default=2.0, help="Min r")
    parser.add_argument("--r_max", type=float, default=6.5, help="Max r")
    parser.add_argument("--r_num", type=int, default=50, help="Grid resolution for r")
    parser.add_argument("--eps_min", type=float, default=0.0, help="Min coupling eps")
    parser.add_argument("--eps_max", type=float, default=1.0, help="Max coupling eps")
    parser.add_argument("--eps_num", type=int, default=50, help="Grid resolution for eps")
    parser.add_argument("--save", type=str, default="", help="Save path")
    
    args = parser.parse_args()
    
    rs = np.linspace(args.r_min, args.r_max, args.r_num)
    epsilons = np.linspace(args.eps_min, args.eps_max, args.eps_num)
    
    matrix = np.zeros((args.eps_num, args.r_num))
    
    print(f"Generating Lyapunov Diagram (r: [{args.r_min}, {args.r_max}], eps: [{args.eps_min}, {args.eps_max}])...")
    for j, r in enumerate(rs):
        for i, eps in enumerate(epsilons):
            matrix[i, j] = compute_diag_le(args.n, args.rule, r, eps,10000,1000)
        if j % 5 == 0: print(f"Progress: {100*(j+1)/args.r_num:.1f}%")

    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, extent=[args.r_min, args.r_max, args.eps_min, args.eps_max], 
                    origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(im, label='Max Lyapunov Exponent')
    plt.title(f"2D Lyapunov Diagram (r vs Coupling, n={args.n}, Rule={args.rule})")
    plt.xlabel("Parameter r")
    plt.ylabel("Coupling Strength (epsilon)")
    
    # Highlight the actual model line (eps=1)
    plt.axhline(1.0, color='cyan', linestyle='--', alpha=0.5, label='Actual Model')
    plt.legend()
    
    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Diagram saved to {args.save}")
    else:
        print("Displaying plot (if no window appears, use --save to output a PNG file)...")
        plt.show()
