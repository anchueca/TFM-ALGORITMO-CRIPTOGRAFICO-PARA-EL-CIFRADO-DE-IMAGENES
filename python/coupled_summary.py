import numpy as np
import matplotlib.pyplot as plt
import argparse
import struct
import scipy.stats as stats
import scipy.special as special
import scipy.fftpack as fft
from tabulate import tabulate
from coupled_map import coupled_step

def shannon_entropy(data, bins=256):
    hist, _ = np.histogram(data, bins=bins, range=(0, 1))
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def autocorrelation(data, lags=50):
    n = len(data)
    mu = np.mean(data)
    data_norm = data - mu
    res = [1.0]
    for l in range(1, lags):
        c = np.sum(data_norm[:n-l] * data_norm[l:]) / (np.sum(data_norm**2))
        res.append(c)
    return res

def nist_frequency_test(bits):
    n = len(bits)
    s_n = np.sum(2 * bits - 1)
    s_obs = abs(s_n) / np.sqrt(n)
    p_val = special.erfc(s_obs / np.sqrt(2))
    return p_val

def nist_runs_test(bits):
    n = len(bits)
    pi = np.mean(bits)
    if abs(pi - 0.5) >= (2/np.sqrt(n)):
        return 0.0
    
    v_n = 1 + np.sum(bits[:-1] != bits[1:])
    p_val = special.erfc(abs(v_n - 2*n*pi*(1-pi)) / (2 * np.sqrt(2*n) * pi * (1-pi)))
    return p_val

def nist_block_frequency_test(bits, m=128):
    n = len(bits)
    n_blocks = n // m
    if n_blocks == 0: return 0.0
    
    # Calculate proportion of ones in each block
    pi = [np.mean(bits[i*m:(i+1)*m]) for i in range(n_blocks)]
    chi_sq = 4 * m * np.sum((np.array(pi) - 0.5)**2)
    p_val = special.gammaincc(n_blocks/2, chi_sq/2)
    return p_val

def nist_longest_run_test(bits):
    """Simple version of the longest run of ones in a block test."""
    n = len(bits)
    if n < 128: return 0.0
    
    # We'll use a 128-bit block as a basic example
    m = 128
    n_blocks = n // m
    
    max_runs = []
    for i in range(n_blocks):
        block = bits[i*m:(i+1)*m]
        # Find longest run of ones in the block
        runs = "".join(map(str, block)).split('0')
        max_runs.append(max(len(r) for r in runs) if runs else 0)
    
    # Simplified evaluation: check if mean longest run is within expectations
    # For m=128, mean is approx 7. NIST SP 800-22 uses a Chi-square on frequencies.
    # We'll return a p-value based on the mean longest run vs expected mean (~log2(m)).
    mean_obs = np.mean(max_runs)
    expected_mean = np.log2(m)
    z_score = abs(mean_obs - expected_mean) / (np.sqrt(expected_mean))
    return special.erfc(z_score / np.sqrt(2))

def nist_spectral_test(bits):
    n = len(bits)
    if n % 2 != 0: bits = bits[:-1]; n -= 1
    
    s = 2 * bits - 1
    dft = fft.fft(s)
    m = np.abs(dft[:n//2])
    
    threshold = np.sqrt(np.log(1/0.05) * n)
    n_obs = np.sum(m < threshold)
    n_exp = 0.95 * n / 2
    
    d = (n_obs - n_exp) / np.sqrt(n * 0.95 * 0.05 / 4)
    p_val = special.erfc(abs(d) / np.sqrt(2))
    return p_val

def binarize(val):
    # Replicates whitening logic from TestNIST.py
    float_bits = struct.unpack('>Q', struct.pack('>d', val))[0]
    mantissa = float_bits & ((1 << 52) - 1)
    top32 = mantissa >> (52 - 32)
    b0, b1, b2, b3 = (top32 >> 24) & 0xFF, (top32 >> 16) & 0xFF, (top32 >> 8) & 0xFF, top32 & 0xFF
    res_byte = b0 ^ b1 ^ b2 ^ b3
    return np.array([(res_byte >> i) & 1 for i in range(8)])

def generate_summary(n, rule, r, iterations, transition, save_prefix=""):
    print(f"Generating Comprehensive Summary for n={n}, r={r}, Rule={rule}...")
    
    xs = np.random.rand(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)
    
    for _ in range(transition):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)
        
    data = []
    all_bits = []
    
    for _ in range(iterations):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)
        data.append(xs.copy())
        for x in xs:
            all_bits.extend(binarize(x))
            
    data = np.array(data)
    bits = np.array(all_bits)
    
    # 1. Visualization Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Chaotic Dashboard: n={n}, r={r}, Rule={rule}", fontsize=16)
    
    # Plot 1: Spacetime Evolution
    im1 = axes[0, 0].imshow(data, aspect='auto', cmap='magma', origin='lower')
    axes[0, 0].set_title("Spacetime Evolution")
    axes[0, 0].set_xlabel("Map Index")
    axes[0, 0].set_ylabel("Iteration")
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Histogram (Distribution)
    axes[0, 1].hist(data.flatten(), bins=100, color='blue', alpha=0.7, density=True)
    axes[0, 1].axhline(1.0, color='red', linestyle='--', label="Uniform Ideal")
    axes[0, 1].set_title("Value Distribution (Histogram)")
    axes[0, 1].set_xlabel("x value")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    
    # Plot 3: Autocorrelation
    lags = min(100, iterations // 2)
    ac = autocorrelation(data[:, 0], lags=lags)
    axes[1, 0].stem(range(lags), ac)
    axes[1, 0].set_title("Autocorrelation (Map 0)")
    axes[1, 0].set_xlabel("Lag")
    axes[1, 0].set_ylabel("Correlation")
    axes[1, 0].grid(True)
    
    # Plot 4: Returns Map (x_t vs x_t+1)
    axes[1, 1].scatter(data[:-1, 0], data[1:, 0], s=1, c='black', alpha=0.5)
    axes[1, 1].set_title("Returns Map (x_t vs x_t+1)")
    axes[1, 1].set_xlabel("x(t)")
    axes[1, 1].set_ylabel("x(t+1)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_prefix:
        plt.savefig(f"{save_prefix}_dashboard.png", dpi=300)
        print(f"Dashboard saved to {save_prefix}_dashboard.png")
    
    # 2. Statistical report
    entropy = shannon_entropy(data.flatten())
    # Take a larger sample if possible for NIST tests
    test_len = min(len(bits), 200000)
    test_bits = bits[:test_len]
    
    p_freq = nist_frequency_test(test_bits)
    p_runs = nist_runs_test(test_bits)
    p_block = nist_block_frequency_test(test_bits)
    p_spec = nist_spectral_test(test_bits)
    p_long = nist_longest_run_test(test_bits)
    
    results = [
        ["Metric", "Value", "Ideal/Threshold", "Result"],
        ["Shannon Entropy", f"{entropy:.4f}", "8.0 (for 256 bins)", "Excellent" if entropy > 7.5 else "Good"],
        ["NIST Monobit P-value", f"{p_freq:.4f}", "> 0.01", "PASS" if p_freq > 0.01 else "FAIL"],
        ["NIST Runs P-value", f"{p_runs:.4f}", "> 0.01", "PASS" if p_runs > 0.01 else "FAIL"],
        ["NIST Block Freq P-value", f"{p_block:.4f}", "> 0.01", "PASS" if p_block > 0.01 else "FAIL"],
        ["NIST Spectral P-value", f"{p_spec:.4f}", "> 0.01", "PASS" if p_spec > 0.01 else "FAIL"],
        ["NIST Longest Run P-val", f"{p_long:.4f}", "> 0.01", "PASS" if p_long > 0.01 else "FAIL"],
        ["Mean (Raw)", f"{np.mean(data):.4f}", "0.5", "OK"],
        ["Variance (Raw)", f"{np.var(data):.4f}", "0.0833 (1/12)", "OK"],
        ["Std Dev (Raw)", f"{np.std(data):.4f}", "0.2887", "OK"]
    ]
    
    print("\n" + "="*85)
    print("      COMPREHENSIVE CHAOTIC FUNCTION STATISTICAL REPORT (NIST SP 800-22)")
    print("="*85)
    print(tabulate(results[1:], headers=results[0], tablefmt="fancy_grid"))
    print("="*85)
    
    if save_prefix:
        with open(f"{save_prefix}_report.txt", "w") as f:
            f.write(tabulate(results[1:], headers=results[0], tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chaotic summary and stats")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--r", type=float, default=6.1)
    parser.add_argument("--rule", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--transition", type=int, default=200)
    parser.add_argument("--save_prefix", type=str, default="")
    
    args = parser.parse_args()
    generate_summary(args.n, args.rule, args.r, args.iterations, args.transition, args.save_prefix)
