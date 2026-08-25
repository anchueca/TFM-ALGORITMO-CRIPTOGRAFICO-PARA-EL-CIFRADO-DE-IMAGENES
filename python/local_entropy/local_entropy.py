#!/usr/bin/env python3
"""
local_entropy.py — Local Shannon Entropy (LSE) Test per Wu et al. (2013)
========================================================================

Evaluates the randomness of cipher images using Local Shannon Entropy over k=30
randomly chosen non-overlapping blocks of n=1936 pixels (44x44 patch).

Statistical Decision Rule:
  Theoretical mean  (μ_k) = 7.902486
  Theoretical var   (σ_k^2) = 0.0000567
  Confidence bounds:
    α = 0.05  => [7.8976, 7.9073]
    α = 0.01  => [7.8961, 7.9089]
    α = 0.001 => [7.8943, 7.9107]
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


def shannon_entropy(data_block):
    """Calculates Shannon Entropy on a 1D or 2D array of uint8 values."""
    flat = data_block.flatten()
    counts = np.bincount(flat, minlength=256)
    probs = counts / float(len(flat))
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def calculate_local_shannon_entropy(image, k=30, n=1936):
    """
    Computes Local Shannon Entropy (LSE) over k non-overlapping blocks of size n pixels.
    For n=1936, patch_side = 44.
    """
    if len(image.shape) == 3:
        # Convert BGR/RGB to grayscale or process channel
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape
    patch_dim = int(np.sqrt(n))

    max_blocks_h = h // patch_dim
    max_blocks_w = w // patch_dim
    total_available = max_blocks_h * max_blocks_w

    if total_available < k:
        # Resize image if too small to fit k blocks of size patch_dim x patch_dim
        scale = np.sqrt(k / total_available) + 0.1
        gray = cv2.resize(gray, (int(w * scale) + patch_dim * 2, int(h * scale) + patch_dim * 2))
        h, w = gray.shape
        max_blocks_h = h // patch_dim
        max_blocks_w = w // patch_dim

    grid_indices = [(r, c) for r in range(max_blocks_h) for c in range(max_blocks_w)]
    np.random.seed(42)  # Reproducible block sampling
    selected_indices = np.random.choice(len(grid_indices), size=k, replace=False)

    entropies = []
    heatmap_matrix = np.full((max_blocks_h, max_blocks_w), np.nan)

    for idx in selected_indices:
        r, c = grid_indices[idx]
        block = gray[r * patch_dim:(r + 1) * patch_dim, c * patch_dim:(c + 1) * patch_dim]
        ent = shannon_entropy(block)
        entropies.append(ent)
        heatmap_matrix[r, c] = ent

    mean_lse = float(np.mean(entropies))
    std_lse = float(np.std(entropies))
    return mean_lse, std_lse, entropies, heatmap_matrix


def evaluate_lse_hypothesis(mean_lse, k=30):
    """
    Performs Wu et al. hypothesis test for LSE.
    Theoretical values for k=30, n=1936:
      μ = 7.902486, σ^2 = 0.0000567 => σ = 0.00753
    """
    mu_k = 7.902486
    sigma_k = 0.00753  # sqrt(0.0000567)

    # Standard normal Z-score
    z_score = (mean_lse - mu_k) / (sigma_k / np.sqrt(k))
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    thresholds = {
        0.05: (7.8976, 7.9073),
        0.01: (7.8961, 7.9089),
        0.001: (7.8943, 7.9107)
    }

    results = {}
    for alpha, (low, high) in thresholds.items():
        passed = (low <= mean_lse <= high)
        results[alpha] = {'passed': passed, 'bounds': (low, high)}

    return p_value, z_score, results


def generate_synthetic_cipher_image(h=512, w=512):
    """Generates synthetic uniform random uint8 image for demonstration."""
    return np.random.randint(0, 256, size=(h, w), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Test de Entropía Local de Shannon (LSE) — Estándar Wu et al. (2013)")
    parser.add_argument("--image", type=str, default="", help="Ruta de la imagen cifrada (PNG/BMP/JPG). Si se omite, genera imagen aleatoria.")
    parser.add_argument("--k", type=int, default=30, help="Número de bloques aleatorios (default: 30)")
    parser.add_argument("--n", type=int, default=1936, help="Número de píxeles por bloque (default: 1936 -> 44x44)")
    parser.add_argument("--save", type=str, default="", help="Ruta de guardado de gráfico PNG (opcional)")

    args = parser.parse_args()

    if args.image and os.path.exists(args.image):
        img = cv2.imread(args.image)
        img_name = os.path.basename(args.image)
    else:
        print("[!] No se proporcionó una imagen válida. Generando imagen sintética aleatoria de 512x512...")
        img = generate_synthetic_cipher_image(512, 512)
        img_name = "Imagen Cifrada Sintética"

    mean_lse, std_lse, entropies, heatmap = calculate_local_shannon_entropy(img, args.k, args.n)
    p_val, z_score, decision_map = evaluate_lse_hypothesis(mean_lse, args.k)

    print("\n" + "=" * 65)
    print(f"RESULTADOS DEL TEST DE ENTROPÍA LOCAL DE SHANNON (LSE): {img_name}")
    print("=" * 65)
    print(f"  Bloques analizados (k)      : {args.k}")
    print(f"  Píxeles por bloque (n)     : {args.n} (44x44)")
    print(f"  Entropía Local Media (h_bar): {mean_lse:.6f}")
    print(f"  Desviación Estándar (σ)    : {std_lse:.6f}")
    print(f"  Z-Score (vs μ=7.902486)     : {z_score:.4f}")
    print(f"  p-value (dos colas)         : {p_val:.6f}")
    print("-" * 65)
    print("  EVALUACIÓN DE HIPÓTESIS DE RANDOMICIDAD:")
    for alpha, res in decision_map.items():
        status = "PASSED [✓]" if res['passed'] else "FAILED [✗]"
        low, high = res['bounds']
        print(f"    α = {alpha:<5}: Intervalo [{low:.4f}, {high:.4f}]  ->  {status}")
    print("=" * 65 + "\n")

    # Plot spatial heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Análisis de Entropía Local de Shannon (LSE) — {img_name}", fontsize=14, fontweight='bold')

    if len(img.shape) == 3:
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(img, cmap='gray')
    ax1.set_title("Imagen Cifrada")
    ax1.axis('off')

    im = ax2.imshow(heatmap, cmap='viridis', origin='upper')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Entropía del Bloque (bits)")
    ax2.set_title(f"Mapa de Calor LSE (k={args.k} bloques, h_bar={mean_lse:.4f})")
    ax2.set_xlabel("Índice de Bloque Horizontal")
    ax2.set_ylabel("Índice de Bloque Vertical")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Gráfico LSE guardado en: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
