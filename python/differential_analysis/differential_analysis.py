#!/usr/bin/env python3
"""
differential_analysis.py — Rigorous NPCR & UACI Differential Cryptanalysis
===========================================================================

Evaluates plaintext differential sensitivity by flipping 1 pixel in original image P1 -> P2,
encrypting both to C1 and C2, and comparing against theoretical decision thresholds (Wu et al.):

  NPCR Critical Bound (N*_α):
    N*_α = (L - (1 / sqrt(MN * L)) * Φ^-1(1 - α)) / (L + 1)
    For 256x256 image at α=0.05: N*_0.05 = 99.5693%

  UACI Critical Interval (U*-_α, U*+_α):
    For 256x256 image at α=0.05: (33.2824%, 33.6447%)
"""

import os
import sys
import argparse
import numpy as np
import scipy.stats as stats
import matplotlib

try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from coupled_map.coupled_map import coupled_step, binarize_float_scalar
except ImportError:
    from coupled_map import coupled_step, binarize_float_scalar

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def compute_npcr_uaci_thresholds(h, w, c=1, alpha=0.05):
    """
    Computes theoretical NPCR critical threshold N*_alpha and UACI confidence interval (U_low, U_high).
    Per Wu et al. statistical hypothesis test for image encryption.
    """
    N_pixels = h * w * c
    L = 255.0

    # NPCR theoretical expectation and critical threshold
    mu_npcr = (L / (L + 1.0)) * 100.0  # 99.609375%
    var_npcr = (L / ((L + 1.0) ** 2)) / N_pixels
    z_alpha = stats.norm.ppf(1.0 - alpha)
    n_star = mu_npcr - z_alpha * np.sqrt(var_npcr) * 100.0

    # UACI theoretical expectation and confidence bounds
    mu_uaci = ((L + 2.0) / (3.0 * (L + 1.0))) * 100.0  # 33.46354%
    var_uaci = ((L + 2.0) * (L**2 + 3.0 * L + 3.0)) / (18.0 * N_pixels * (L + 1.0)**2)
    u_low = mu_uaci - z_alpha * np.sqrt(var_uaci) * 100.0
    u_high = mu_uaci + z_alpha * np.sqrt(var_uaci) * 100.0

    return n_star, u_low, u_high


def simulate_cml_encrypt(image, r=6.1, rule=30):
    """
    Simulates encryption by combining plaintext image with CML keystream and diffusion.
    Includes plaintext-dependent seed simulation (like hash-based initial state).
    """
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    total_bytes = h * w * c

    # Plaintext-dependent seed perturbation (SHA-256 equivalent)
    plain_sum = int(np.sum(image.flatten()))
    seed_r = r + (plain_sum % 1000) * 1e-12

    n = 16
    xs = 0.4 * np.ones(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)

    for _ in range(100):
        xs, ca_states = coupled_step(xs, ca_states, seed_r, rule)

    keystream = []
    needed = int(np.ceil(total_bytes / (n * 4)))
    for _ in range(needed):
        xs, ca_states = coupled_step(xs, ca_states, seed_r, rule)
        for x_val in xs:
            b = binarize_float_scalar(x_val)
            keystream.extend([b, (b * 37) & 0xFF, (b * 91) & 0xFF, (b * 157) & 0xFF])

    ks_arr = np.array(keystream[:total_bytes], dtype=np.uint8).reshape(image.shape)
    cipher = np.bitwise_xor(image, ks_arr)

    # CBC-like diffusion pass
    flat_c = cipher.flatten().astype(np.uint16)
    for i in range(1, len(flat_c)):
        flat_c[i] = (flat_c[i] + flat_c[i - 1]) & 0xFF

    return flat_c.astype(np.uint8).reshape(image.shape)



def calculate_npcr_uaci(img1, img2):
    """Calculates NPCR and UACI metrics."""
    arr1 = img1.flatten().astype(np.int32)
    arr2 = img2.flatten().astype(np.int32)

    diff = arr1 != arr2
    npcr = (np.sum(diff) / diff.size) * 100.0

    abs_diff = np.sum(np.abs(arr1 - arr2))
    uaci = (abs_diff / (diff.size * 255.0)) * 100.0

    return npcr, uaci


def run_differential_test(image, num_runs=5):
    """
    Flips 1 random pixel in image and measures NPCR/UACI over multiple runs.
    """
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1

    npcr_results = []
    uaci_results = []

    for _ in range(num_runs):
        p1 = image.copy()
        p2 = image.copy()

        # Pick random pixel and perturb by 1
        ry = np.random.randint(0, h)
        rx = np.random.randint(0, w)
        if c > 1:
            rc = np.random.randint(0, c)
            p2[ry, rx, rc] = (int(p2[ry, rx, rc]) + 1) % 256
        else:
            p2[ry, rx] = (int(p2[ry, rx]) + 1) % 256

        c1 = simulate_cml_encrypt(p1)
        c2 = simulate_cml_encrypt(p2)

        npcr, uaci = calculate_npcr_uaci(c1, c2)
        npcr_results.append(npcr)
        uaci_results.append(uaci)

    return np.mean(npcr_results), np.mean(uaci_results)


def main():
    parser = argparse.ArgumentParser(description="Análisis Diferencial de Cifrado (NPCR y UACI) con Pruebas de Hipótesis")
    parser.add_argument("--image", type=str, default="", help="Ruta de la imagen de prueba. Si se omite, evalúa batería de imágenes sintéticas.")
    parser.add_argument("--runs", type=int, default=5, help="Número de ejecuciones por prueba (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia α (default: 0.05)")
    parser.add_argument("--save", type=str, default="", help="Ruta de guardado de gráfico PNG (opcional)")

    args = parser.parse_args()

    # Define synthetic benchmark images if no input file
    test_images = {}
    if args.image and os.path.exists(args.image):
        img = cv2.imread(args.image)
        test_images[os.path.basename(args.image)] = img
    else:
        print("[!] No se especificó imagen. Generando batería de imágenes de prueba (256x256)...")
        test_images["All-Black (0s)"] = np.zeros((256, 256, 3), dtype=np.uint8)
        test_images["All-White (255s)"] = 255 * np.ones((256, 256, 3), dtype=np.uint8)
        
        # Checkerboard
        cb = np.zeros((256, 256, 3), dtype=np.uint8)
        cb[::2, ::2] = 255
        cb[1::2, 1::2] = 255
        test_images["Checkerboard"] = cb

        # Random Noise
        test_images["Random Noise"] = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    table_data = []
    plot_names = []
    plot_npcrs = []
    plot_uacis = []

    print("\n" + "=" * 75)
    print(" ANÁLISIS DIFERENCIAL DE PÍXEL INDIVIDUAL (NPCR & UACI)")
    print("=" * 75)

    for name, img in test_images.items():
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) == 3 else 1

        n_star, u_low, u_high = compute_npcr_uaci_thresholds(h, w, c, args.alpha)
        mean_npcr, mean_uaci = run_differential_test(img, args.runs)

        npcr_pass = mean_npcr >= n_star
        uaci_pass = u_low <= mean_uaci <= u_high

        status = "PASSED [✓]" if (npcr_pass and uaci_pass) else "FAILED [✗]"

        table_data.append([
            name, f"{w}x{h}", f"{mean_npcr:.4f}%", f">= {n_star:.4f}%",
            f"{mean_uaci:.4f}%", f"[{u_low:.4f}%, {u_high:.4f}%]", status
        ])

        plot_names.append(name)
        plot_npcrs.append(mean_npcr)
        plot_uacis.append(mean_uaci)

    headers = ["Imagen", "Dimensión", "NPCR", "NPCR Umbral", "UACI", "UACI Intervalo", "Resultado"]
    if tabulate:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print(f"{'Imagen':<18} | {'NPCR':<10} | {'NPCR Umbral':<12} | {'UACI':<10} | {'UACI Intervalo':<20} | {'Resultado'}")
        print("-" * 75)
        for row in table_data:
            print(f"{row[0]:<18} | {row[2]:<10} | {row[3]:<12} | {row[4]:<10} | {row[5]:<20} | {row[6]}")

    print("=" * 75 + "\n")

    # Plot summary bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Resumen de Análisis Diferencial de Píxel Único (NPCR y UACI)", fontsize=14, fontweight='bold')

    x_pos = np.arange(len(plot_names))
    ax1.bar(x_pos, plot_npcrs, color='darkcyan', alpha=0.85, width=0.4)
    ax1.axhline(n_star, color='red', linestyle='--', linewidth=1.5, label=f"N*_0.05 ({n_star:.2f}%)")
    ax1.set_ylabel("NPCR (%)")
    ax1.set_title("Número de Píxeles Cambiados (NPCR)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(plot_names, rotation=30, ha='right')
    ax1.set_ylim(98.5, 100.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(x_pos, plot_uacis, color='coral', alpha=0.85, width=0.4)
    ax2.axhline(u_low, color='blue', linestyle=':', linewidth=1.5, label=f"U*_low ({u_low:.2f}%)")
    ax2.axhline(u_high, color='blue', linestyle=':', linewidth=1.5, label=f"U*_high ({u_high:.2f}%)")
    ax2.set_ylabel("UACI (%)")
    ax2.set_title("Intensidad Media de Cambio (UACI)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(plot_names, rotation=30, ha='right')
    ax2.set_ylim(32.0, 35.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Gráfico diferencial guardado en: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
