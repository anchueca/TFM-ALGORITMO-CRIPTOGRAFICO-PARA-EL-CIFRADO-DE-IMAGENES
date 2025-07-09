import cv2
import numpy as np
import argparse
from scipy.stats import entropy, chisquare
from tabulate import tabulate
import matplotlib.pyplot as plt
import subprocess
import os
import sys
import time

from modeloCaos import uno
from proposed_cipher_cuda import encrypt_image, unencrypt_image, unstack_image,generate_password,show_image
from proposed_cipher import encrypt_image as lineal_encrypt_image, unencrypt_image as lineal_unencrypt_image

def plot_histograms(original, ciphered):
    for image, label in zip([original, ciphered], ['Original', 'Cifrada']):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist /= hist.sum()
        plt.figure(figsize=(10, 6))
        plt.bar(range(256), hist[:, 0], color='gray', edgecolor='black')
        plt.title(f'Histograma {label}')
        plt.xlabel('Intensidad')
        plt.ylabel('Frecuencia Normalizada')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

def bit_difference(a, b):
    return bin(a ^ b).count('1')

def compute_bcr(cipher1, cipher2):
    if cipher1.shape != cipher2.shape:
        raise ValueError("Las imágenes deben tener el mismo tamaño")
    
    if len(cipher1.shape) == 3 and cipher1.shape[2] == 1:
        cipher1 = cipher1[:, :, 0]
        cipher2 = cipher2[:, :, 0]

    h, w = cipher1.shape
    total_bits = h * w * 8
    total_diff_bits = 0

    for i in range(h):
        for j in range(w):
            total_diff_bits += bin(cipher1[i, j] ^ cipher2[i, j]).count('1')

    bcr = (total_diff_bits / total_bits) * 100
    return bcr


def compute_sdr(cipher1, cipher2):
    if cipher1.shape != cipher2.shape:
        raise ValueError("Las imágenes deben tener el mismo tamaño")

    diff = np.bitwise_xor(cipher1, cipher2)
    bit_changes = np.zeros(8)

    # contar cambios de bits por posición
    for bit in range(8):
        mask = 1 << bit
        bit_changes[bit] = np.sum((diff & mask) > 0)

    total_changes = np.sum(bit_changes)
    if total_changes == 0:
        print("[!] No hay cambios de bits. SDR = 0")
        return 0.0

    probs = bit_changes / total_changes
    sdr = -np.sum(probs * np.log2(probs + 1e-12))  # evitamos log(0)
    
    return sdr

def calculate_npc(image1, image2):
    diff = image1 != image2
    npcr = np.sum(diff) / diff.size * 100
    return npcr

def calculate_uaci(image1, image2):
    diff = np.abs(image1.astype(np.int16) - image2.astype(np.int16))
    uaci = np.sum(diff) / (diff.size * 255) * 100
    return uaci

def calculate_cc(image1, image2):
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    correlation_matrix = np.corrcoef(image1_flat, image2_flat)
    return correlation_matrix[0, 1]

def calculate_ie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    prob = hist / np.sum(hist)
    return entropy(prob.ravel() + 1e-9, base=2)

def calculate_mse(image1, image2):
    err = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
    return err

def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_directional_cc(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(image.shape) > 2 else image.astype(np.float32)
    x_h = gray[:, :-1].flatten()
    y_h = gray[:, 1:].flatten()
    cc_h = np.corrcoef(x_h, y_h)[0, 1]
    x_v = gray[:-1, :].flatten()
    y_v = gray[1:, :].flatten()
    cc_v = np.corrcoef(x_v, y_v)[0, 1]
    x_d = gray[:-1, :-1].flatten()
    y_d = gray[1:, 1:].flatten()
    cc_d = np.corrcoef(x_d, y_d)[0, 1]
    return cc_h, cc_v, cc_d

def chi_square_test(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    expected = np.ones_like(hist) * np.sum(hist) / 256
    chi2, p_value = chisquare(hist, expected)
    return chi2, p_value

def encrypt_image_external_cypher(input_image: str, output_image: str, password:str, cipher_program_path:str, *args):
    command = [cipher_program_path, input_image, output_image, password] + list(args)
    print(f"Ejecutando comando: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error al ejecutar el cifrado: {result.stderr}")
        return None
    image = cv2.imread(output_image)
    if input_image != output_image:
        os.remove(output_image)
    return image

def autocorrelation_2d(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(image.shape) > 2 else image.astype(np.float32)
    mean = np.mean(gray)
    norm = gray - mean
    corr = np.sum(norm * np.roll(norm, 1, axis=0)) / np.sum(norm ** 2)
    return corr

def conditional_entropy_rgb(image):
    r, g, b = cv2.split(image)
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
    r_prob = r_hist / np.sum(r_hist)
    g_prob = g_hist / np.sum(g_hist)
    b_prob = b_hist / np.sum(b_hist)
    return entropy(r_prob.ravel() + 1e-9), entropy(g_prob.ravel() + 1e-9), entropy(b_prob.ravel() + 1e-9)

def bit_change_rate(image1, image2):
    diff_bits = np.unpackbits(np.bitwise_xor(image1, image2)).sum()
    total_bits = image1.size * 8
    return diff_bits / total_bits * 100

def nist_monobit_test(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    bits = np.unpackbits(gray)
    ones = np.sum(bits)
    zeros = bits.size - ones
    return ones / bits.size, zeros / bits.size

def message_sensitivity_test_external(image_path, password, cipher_program_path, *args):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {image_path}.")
        return None, None
    altered = image.copy()
    altered[0, 0, 0] ^= 1
    input_filename = os.path.basename(image_path)
    output_filename_original = "ciphered_" + input_filename
    output_filename_altered = "ciphered_" + "altered_" + input_filename
    cv2.imwrite(output_filename_altered, altered)
    cipher1 = encrypt_image_external_cypher(image_path, output_filename_original, password, cipher_program_path, *args)
    cipher2 = encrypt_image_external_cypher(output_filename_altered, output_filename_altered, password, cipher_program_path, *args)
    os.remove(output_filename_altered)
    if cipher1 is None or cipher2 is None:
        print("Error en el proceso de cifrado.")
        return None, None
    npcr = calculate_npc(cipher1, cipher2)
    uaci = calculate_uaci(cipher1, cipher2)
    return npcr, uaci

def message_sensitivity_test(image, model, password, rounds, *args):
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {image}.")
        return None, None
    altered = image.copy()
    altered[0, 0, 0] ^= 1
    cipher1 = encrypt_image(image, model, password, rounds, *args)
    cipher2 = encrypt_image(altered, model, password, rounds, *args)
    if cipher1 is None or cipher2 is None:
        print("Error en el proceso de cifrado.")
        return None, None
    npcr = calculate_npc(cipher1, cipher2)
    uaci = calculate_uaci(cipher1, cipher2)
    return npcr, uaci

def flip_one_bit_in_key(key): # ad hoc
    if isinstance(key, list):
        if isinstance(key[0],float):
            key[0] + .000000001
        else: 
            if isinstance(key,list):
                key[0][0] + .000000001

    else: key ^ 1

    return key
    

def test_key_sensitivity(image, ciphered, key_original):
    """
    Prueba la sensibilidad del cifrado a pequeñas modificaciones en la clave.
    """

    num_blocks= 256
    precission_level=2
    image_rows,image_columns = image.shape[:2]
    rounds=3

    password = generate_password(key_original,num_blocks,image_rows,image_columns,precission_level,rounds)
    password2 = generate_password("ñsñsdsd",num_blocks,image_rows,image_columns,precission_level,rounds)
    labels= ['row_password', 'column_password', 'block_password', 'flow_password']

    for i in range(len(labels)):
        actual_password= list(password)
        actual_password[i] = password2[i]#flip_one_bit_in_key(actual_password[i])
        cipher_modified = unencrypt_image(ciphered, password)
        #unencrypt_image(ciphered, actual_password)
        
        print("[Test de Sensibilidad a la Clave", labels[i],"]")
        compute_comparison_statistics(image, cipher_modified)
        show_image(cipher_modified)

def compute_single_image_statistics(image):
    print(f"Estadísticas de la imagen:")
    ie = calculate_ie(image)
    cc_h, cc_v, cc_d = calculate_directional_cc(image)
    chi2, p_chi = chi_square_test(image)

    headers = ["Estadística", "Valor"]
    table = [
        ["IE", f"{ie:.4f} bits"],
        ["CC-Horiz", f"{cc_h:.4f}"],
        ["CC-Vert", f"{cc_v:.4f}"],
        ["CC-Diag", f"{cc_d:.4f}"],
        ["Chi²", f"{chi2:.2f}"],
        ["p chi", f"{p_chi:.2f}"]
    ]
    print(tabulate(table, headers=headers, tablefmt="pretty"))

def compute_comparison_statistics(original, ciphered):
    if original.shape != ciphered.shape:
        print("Error: Las imágenes deben tener el mismo tamaño y número de canales.")
        return

    npcr = calculate_npc(original, ciphered)
    uaci = calculate_uaci(original, ciphered)
    cc = calculate_cc(original, ciphered)
    mse = calculate_mse(original, ciphered)
    psnr = calculate_psnr(original, ciphered)
    bcr = compute_bcr(original, ciphered)
    sdr = compute_sdr(original, ciphered)

    print("\n[Comparación entre imágenes]")
    headers = ["Estadística", "Valor"]
    table = [
        ["NPCR", f"{npcr:.4f}%"],
        ["UACI", f"{uaci:.4f}%"],
        ["CC-global", f"{cc:.4f}"],
        ["MSE", f"{mse:.4f}"],
        ["PSNR", f"{psnr:.4f} dB"],
        ["BCR", f"{bcr:.4f}%"],
        ["SDR", f"{sdr:.4f} bits"],
    ]
    print(tabulate(table, headers=headers, tablefmt="pretty"))

def main():
    parser = argparse.ArgumentParser(description="Cifrado de imágenes con CUDA y modelos caóticos")
    parser.add_argument("input_image", help="Ruta de la imagen de entrada")
    parser.add_argument("password", help="Contraseña para el cifrado")
    parser.add_argument("--rounds", type=int, default=3, help="Número de rondas de cifrado")
    args = parser.parse_args()

    image_path = args.input_image
    password = args.password
    rounds = args.rounds

    print(image_path)
    original = cv2.imread(image_path)
    if original is None:
        print("No se pudo cargar la imagen.")
        return
    
    if len(original.shape) > 2:
        original = unstack_image(original)

    print("[+] Iniciando cifrado...")
    start_enc = time.time()
    ciphered = encrypt_image(original, password, rounds)
    end_enc = time.time()
    print(f"[+] Cifrado completado en {end_enc - start_enc:.4f} segundos")

    print("[+] Iniciando descifrado...")
    start_dec = time.time()
    deciphered = unencrypt_image(ciphered, password, rounds)
    end_dec = time.time()
    print(f"[+] Descifrado completado en {end_dec - start_dec:.4f} segundos")

    print("[+] Iniciando cifrado lineal...")
    start_enc = time.time()
    ciphered = lineal_encrypt_image(original,uno, password, rounds)
    end_enc = time.time()
    print(f"[+] Cifrado completado lineal en {end_enc - start_enc:.4f} segundos")

    print("[+] Iniciando descifrado lineal...")
    start_dec = time.time()
    deciphered = lineal_unencrypt_image(ciphered,uno, password, rounds)
    end_dec = time.time()
    print(f"[+] Descifrado lineal completado en {end_dec - start_dec:.4f} segundos")

    # Guardar imagen cifrada y descifrada
    cv2.imwrite("ciphered_output.png", ciphered)
    #cv2.imwrite("deciphered_output.png", deciphered)

    # Visualizar histogramas
    #plot_histograms(original, ciphered)

    # Mostrar métricas estadísticas
    #compute_single_image_statistics(original)
    #compute_single_image_statistics(ciphered)
    ciphered2 = encrypt_image(original, 2*password, rounds)
    #compute_comparison_statistics(ciphered, ciphered2)
    #test_key_sensitivity(original,ciphered, password)


if __name__ == "__main__":
    main()