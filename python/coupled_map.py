import numpy as np

def cosine_cosine_map(x, r):
    """
    Chaotic function: |cos(pi * r * cos(pi * t) * t)| where t = r + 3*x^2
    Matches CUDA implementation in kernels.cuh
    """
    t = r + 3.0 * x * x
    return np.abs(np.cos(np.pi * r * np.cos(np.pi * t) * t))

def evolve_ca_16bit(state, rule=30):
    """
    Evolves a 16-bit CA state using bitwise operations.
    Matches CUDA evolve_16bit_isolated in automataKernel.cuh
    """
    # Periodic boundary wrap-around for 16 bits
    L = ((state >> 1) | (state << 15)) & 0xFFFF
    R = ((state << 1) | (state >> 15)) & 0xFFFF
    C = state & 0xFFFF
    
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
    
    xs: array of n chaotic variables
    ca_states: array of n 16-bit CA states
    r: chaotic parameter
    """
    n = len(xs)
    # 1. Individual chaotic evolution (map)
    mapped_xs = cosine_cosine_map(xs, r)
    
    # 2. CA evolution
    new_ca_states = np.array([evolve_ca_16bit(int(s), rule) for s in ca_states], dtype=np.uint16)
    
    # 3. Coupling (Weighted average with neighbors in a ring)
    new_xs = np.zeros_like(xs)
    for i in range(n):
        # Indices for neighbors in a ring
        idx_prev = (i - 1) % n
        idx_next = (i + 1) % n
        
        # Weights from CA state
        # High 8 bits for original proportionality, low 8 bits for neighbor proportionality
        evolved = int(new_ca_states[i])
        v1 = ((evolved >> 8) & 0xFF) / 255.0
        v2 = (evolved & 0xFF) / 255.0
        
        c_influence = v1
        rest = 1.0 - v1
        r_influence = rest * v2
        l_influence = rest * (1.0 - v2)
        
        # Mixed value: weighted average of i-th mapped value and its neighbors
        # Note: In CUDA, the neighbors are from the previous iteration or current?
        # In keysream_generation_parallel: 
        # _cr = coupled_map(next_val, r_seed, l_seed, cellular_automata_value);
        # where next_val is mapped(c_seed), r_seed/l_seed are neighbors.
        
        new_xs[i] = (mapped_xs[i] * c_influence) + \
                     (mapped_xs[idx_next] * r_influence) + \
                     (mapped_xs[idx_prev] * l_influence)
                     
    return new_xs, new_ca_states
