import numpy as np

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
    # Periodic boundary wrap-around for 16 bits
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
    
    xs: array of n chaotic variables
    ca_states: array of n 16-bit CA states
    r: chaotic parameter
    """
    n = len(xs)
    # 1. Individual chaotic evolution (map)
    mapped_xs = cosine_cosine_map(xs, r)
    
    # 2. CA evolution and Coupling (Weighted average with neighbors in a ring)
    new_ca_states = np.zeros_like(ca_states, dtype=np.uint16)
    new_xs = np.zeros_like(xs)
    
    for i in range(n):
        # Indices for neighbors in a ring
        idx_prev = (i - 1) % n
        idx_next = (i + 1) % n
        
        # Evolve CA
        evolved = evolve_ca_16bit(int(ca_states[i]), rule)
        
        # Weights from CA state
        # High 8 bits for original proportionality, low 8 bits for neighbor proportionality
        v1 = ((evolved >> 8) & 0xFF) / 255.0
        v2 = (evolved & 0xFF) / 255.0
        
        c_influence = v1
        rest = 1.0 - v1
        r_influence = rest * v2
        l_influence = rest * (1.0 - v2)
        
        new_xs[i] = (mapped_xs[i] * c_influence) + \
                     (mapped_xs[idx_next] * r_influence) + \
                     (mapped_xs[idx_prev] * l_influence)
                     
        # Bidirectional coupling: perturb CA state using chaotic output
        noise = binarize_float_scalar(mapped_xs[i])
        noise16 = (noise << 8) | noise
        new_ca_states[i] = (evolved ^ noise16) & 0xFFFF
        
    return new_xs, new_ca_states
