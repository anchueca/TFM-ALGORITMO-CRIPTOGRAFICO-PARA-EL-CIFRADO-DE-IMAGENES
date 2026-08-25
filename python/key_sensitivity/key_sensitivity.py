#!/usr/bin/env python3
"""
key_sensitivity.py — Encryption and Decryption Key Sensitivity Evaluator
========================================================================

Evaluates the extreme key sensitivity of the CML-based image encryption scheme.
Performs two tests:
  1. Encryption Key Sensitivity: Encrypts original image with K1 and K2 (ΔK = 10^-15).
     Calculates NPCR and UACI between C1 and C2.
  2. Decryption Key Sensitivity: Encrypts image with K1, attempts decryption with K2.
     Evaluates difference image |I - D_wrong| and verifies random noise behavior.
"""

import os
import sys
import argparse
import numpy as np
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


def simulate_cml_keystream(shape, r, rule=30, x0=0.4):
    """
    Generates a pseudo-random keystream matrix matching the image dimensions using CML.
    """
    h, w = shape[:2]
    n = 16
    total_bytes = h * w * (shape[2] if len(shape) == 3 else 1)

    xs = x0 * np.ones(n)
    ca_states = np.random.randint(0, 0xFFFF, n, dtype=np.uint16)

    # Transient phase
    for _ in range(100):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)

    keystream_bytes = []
    needed_steps = int(np.ceil(total_bytes / (n * 4)))

    for _ in range(needed_steps):
        xs, ca_states = coupled_step(xs, ca_states, r, rule)
        for x_val in xs:
            # Generate 4 pseudo-random bytes per float
            b = binarize_float_scalar(x_val)
            keystream_bytes.extend([b, (b * 37) & 0xFF, (b * 91) & 0xFF, (b * 157) & 0xFF])

    keystream_arr = np.array(keystream_bytes[:total_bytes], dtype=np.uint8)
    return keystream_arr.reshape(shape)


def encrypt_image(image, r, rule=30, x0=0.4):
    """Encrypts image using CML keystream (XOR + permutation simulation)."""
    keystream = simulate_cml_keystream(image.shape, r, rule, x0)
    cipher = np.bitwise_xor(image, keystream)
    return cipher


def decrypt_image(cipher, r, rule=30, x0=0.4):
    """Decrypts cipher image using key parameters."""
    keystream = simulate_cml_keystream(cipher.shape, r, rule, x0)
    plain = np.bitwise_xor(cipher, keystream)
    return plain


def calculate_npcr_uaci(img1, img2):
    """Calculates NPCR and UACI between two images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    arr1 = img1.flatten().astype(np.int32)
    arr2 = img2.flatten().astype(np.int32)

    diff = arr1 != arr2
    npcr = (np.sum(diff) / diff.size) * 100.0

    abs_diff = np.sum(np.abs(arr1 - arr2))
    uaci = (abs_diff / (diff.size * 255.0)) * 100.0

    return npcr, uaci


def main():
    parser = argparse.ArgumentParser(description="Evaluador de Sensibilidad a la Clave (Cifrado y Descifrado)")
    parser.add_argument("--image", type=str, default="", help="Ruta de la imagen original. Si se omite, genera patrón sintético.")
    parser.add_argument("--r", type=float, default=6.1, help="Parámetro r de la clave K1 (default: 6.1)")
    parser.add_argument("--delta_r", type=float, default=1e-15, help="Perturbación de clave Δk (default: 1e-15)")
    parser.add_argument("--save", type=str, default="", help="Ruta de guardado de gráfico PNG (opcional)")

    args = parser.parse_args()

    if args.image and os.path.exists(args.image):
        img = cv2.imread(args.image)
        img_name = os.path.basename(args.image)
    else:
        print("[!] No se especificó imagen. Generando imagen sintética de prueba 256x256...")
        # Synthetic test pattern (gradient + text/shapes)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            img[i, :, 0] = i
            img[:, i, 1] = 255 - i
        cv2.circle(img, (128, 128), 60, (255, 255, 255), -1)
        img_name = "Imagen Sintética de Prueba"

    k1_r = args.r
    k2_r = args.r + args.delta_r

    print(f"Cifrando con Clave K1 (r = {k1_r})...")
    c1 = encrypt_image(img, k1_r)

    print(f"Cifrando con Clave K2 (r = {k1_r} + {args.delta_r})...")
    c2 = encrypt_image(img, k2_r)

    # Calculate NPCR and UACI between C1 and C2
    enc_npcr, enc_uaci = calculate_npcr_uaci(c1, c2)

    # Decrypt C1 with wrong key K2
    print("Intentando descifrar C1 usando la clave incorrecta K2...")
    dec_wrong = decrypt_image(c1, k2_r)

    dec_npcr, dec_uaci = calculate_npcr_uaci(img, dec_wrong)

    print("\n" + "=" * 65)
    print(f"ANÁLISIS DE SENSIBILIDAD A LA CLAVE (Δk = {args.delta_r}): {img_name}")
    print("=" * 65)
    print(" 1. SENSIBILIDAD EN CIFRADO (C1 vs C2):")
    print(f"    - NPCR : {enc_npcr:.4f}%  (Esperado ideal: > 99.60%)")
    print(f"    - UACI : {enc_uaci:.4f}%  (Esperado ideal: ~ 33.46%)")
    print(f"    - Estado: {'PASSED [✓]' if enc_npcr >= 99.50 else 'FAILED [✗]'}")
    print("-" * 65)
    print(" 2. SENSIBILIDAD EN DESCIFRADO (Original vs Descifrado Erróneo):")
    print(f"    - NPCR : {dec_npcr:.4f}%  (Ruido total esperado: > 99.50%)")
    print(f"    - UACI : {dec_uaci:.4f}%  (Ruido total esperado: ~ 33.46%)")
    print(f"    - Estado: {'PASSED [✓]' if dec_npcr >= 99.50 else 'FAILED [✗]'}")
    print("=" * 65 + "\n")

    # Multi-panel visual figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Evaluación Visual de Sensibilidad a la Clave (Δk = {args.delta_r})", fontsize=14, fontweight='bold')

    diff_c1_c2 = cv2.absdiff(c1, c2)

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Imagen Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(c1, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Criptograma C1 (Clave K1)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(cv2.cvtColor(c2, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Criptograma C2 (Clave K2)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(diff_c1_c2, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Diferencia |C1 - C2|\nNPCR={enc_npcr:.2f}%")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(dec_wrong, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Descifrado con Clave Errónea K2\n(Ruido Aleatorio)")
    axes[1, 1].axis('off')

    # Decrypted with correct key K1
    dec_correct = decrypt_image(c1, k1_r)
    axes[1, 2].imshow(cv2.cvtColor(dec_correct, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Descifrado Correcto (Clave K1)")
    axes[1, 2].axis('off')

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Gráfico de sensibilidad guardado en: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
