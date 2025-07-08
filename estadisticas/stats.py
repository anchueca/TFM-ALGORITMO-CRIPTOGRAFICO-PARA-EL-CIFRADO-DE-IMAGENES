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

from proposed_cipher_cuda import encrypt_image,unencrypt_image
from modeloCaos import selectFunction

def plot_histograms(original, ciphered):
    # Definir una función para evitar convertir a escala de grises si ya está
    def convert_to_gray(image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image  # Ya está en escala de grises

    for image, label in zip([original, ciphered], ['Original', 'Cifrada']):
        #image = convert_to_gray(image)

        # Calcular el histograma
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Normalizar el histograma
        hist /= hist.sum()

        # Graficar como barras
        plt.figure(figsize=(10, 6))
        plt.bar(range(256), hist[:, 0], color='gray', edgecolor='black')
        plt.title(f'Histograma {label}')
        plt.xlabel('Intensidad')
        plt.ylabel('Frecuencia Normalizada')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


# NPCR: Porcentaje de píxeles distintos entre dos imágenes
def calculate_npc(image1, image2):
    diff = image1 != image2
    npcr = np.sum(diff) / diff.size * 100
    return npcr

# UACI: Promedio del cambio absoluto normalizado entre dos imágenes
def calculate_uaci(image1, image2):
    diff = np.abs(image1.astype(np.int16) - image2.astype(np.int16))
    uaci = np.sum(diff) / (diff.size * 255) * 100
    return uaci

# CC global entre dos imágenes
def calculate_cc(image1, image2):
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    correlation_matrix = np.corrcoef(image1_flat, image2_flat)
    return correlation_matrix[0, 1]

# IE: Entropía de una imagen (aleatoriedad)
def calculate_ie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    prob = hist / np.sum(hist)
    return entropy(prob.ravel() + 1e-9, base=2)

# MSE: Error cuadrático medio
def calculate_mse(image1, image2):
    err = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
    return err

# PSNR: Relación señal/ruido de pico
def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Correlación entre píxeles adyacentes en distintas direcciones
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

# Chi-cuadrado sobre histograma
def chi_square_test(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    expected = np.ones_like(hist) * np.sum(hist) / 256
    chi2, _ = chisquare(hist, expected)
    return chi2

# Función que ejecuta el programa externo de cifrado
def encrypt_image_external_cypher(input_image: str, output_image: str, password:str, cipher_program_path:str, *args):
    command = [
        cipher_program_path,       # Ruta del programa de cifrado
        input_image,               # Nombre de la imagen original
        output_image,              # Nombre de la imagen cifrada
        password                   # Contraseña
    ] + list(args)  # Otros argumentos que pueda necesitar el programa
    print(f"Ejecutando comando: {' '.join(command)}")
    # Ejecutar el comando y esperar a que termine
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error al ejecutar el cifrado: {result.stderr}")
        return None
    
    image=cv2.imread(output_image)
    if(input_image!=output_image):
        os.remove(output_image) # Elimina la imagen cifrada
    
    return image  # Devuelve la imagen cifrada

# Autocorrelación 2D
def autocorrelation_2d(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(image.shape) > 2 else image.astype(np.float32)
    mean = np.mean(gray)
    norm = gray - mean
    corr = np.sum(norm * np.roll(norm, 1, axis=0)) / np.sum(norm ** 2)
    return corr

# Entropía condicional RGB (simple)
def conditional_entropy_rgb(image):
    r, g, b = cv2.split(image)
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
    r_prob = r_hist / np.sum(r_hist)
    g_prob = g_hist / np.sum(g_hist)
    b_prob = b_hist / np.sum(b_hist)
    return entropy(r_prob.ravel() + 1e-9), entropy(g_prob.ravel() + 1e-9), entropy(b_prob.ravel() + 1e-9)

# Tasa de cambio de bits (Bit Change Rate, BCR)
def bit_change_rate(image1, image2):
    diff_bits = np.unpackbits(np.bitwise_xor(image1, image2)).sum()
    total_bits = image1.size * 8
    return diff_bits / total_bits * 100

# NIST Monobit Test (versión simple)
def nist_monobit_test(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    bits = np.unpackbits(gray)
    ones = np.sum(bits)
    zeros = bits.size - ones
    return ones / bits.size, zeros / bits.size

# Prueba de sensibilidad al mensaje (cambio de un bit)
def message_sensitivity_test_external(image_path, password, cipher_program_path, *args):
    # Cargar la imagen desde la ruta
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {image_path}.")
        return None, None

    # Crear una copia de la imagen y modificar un bit
    altered = image.copy()
    altered[0, 0, 0] ^= 1  # Flip un bit del primer píxel
    
    # Generar los nombres de los archivos cifrados
    input_filename = os.path.basename(image_path)
    output_filename_original = "ciphered_" + input_filename
    output_filename_altered = "ciphered_" + "altered_" + input_filename
    cv2.imwrite(output_filename_altered, altered)
    
    # Cifra las dos imágenes
    cipher1 = encrypt_image_external_cypher(image_path, output_filename_original, password, cipher_program_path, *args)
    cipher2 = encrypt_image_external_cypher(output_filename_altered, output_filename_altered, password, cipher_program_path, *args)

    os.remove(output_filename_altered)
    
    if cipher1 is None or cipher2 is None:
        print("Error en el proceso de cifrado.")
        return None, None
    
    # Calcula NPCR y UACI
    npcr = calculate_npc(cipher1, cipher2)
    uaci = calculate_uaci(cipher1, cipher2)
    
    return npcr, uaci


# Prueba de sensibilidad al mensaje (cambio de un bit)
def message_sensitivity_test(image,model, password,rounds, *args):
    
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {image}.")
        return None, None

    # Crear una copia de la imagen y modificar un bit
    altered = image.copy()
    
    altered[0, 0, 0] ^= 1  # Flip un bit del primer píxel
    
    # Cifra las dos imágenes
    cipher1 = encrypt_image(image,model, password, rounds, *args)
    cipher2 = encrypt_image(altered,model, password, rounds, *args)
    
    if cipher1 is None or cipher2 is None:
        print("Error en el proceso de cifrado.")
        return None, None
    
    # Calcula NPCR y UACI
    npcr = calculate_npc(cipher1, cipher2)
    uaci = calculate_uaci(cipher1, cipher2)
    
    return npcr, uaci

# Prueba de sensibilidad al mensaje (cambio de un bit)
def password_sensitivity_test(image,model, password,rounds, *args):
    
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {image}.")
        return None, None

    # Crear una copia de la imagen y modificar un bit
    altered_password = "juan"
    # Cifra las dos imágenes
    cipher1 = encrypt_image(image,model,password, rounds, *args)
    cipher2 = encrypt_image(image,model, altered_password, rounds, *args)
    
    if cipher1 is None or cipher2 is None:
        print("Error en el proceso de cifrado.")
        return None, None
    
    # Calcula NPCR y UACI
    npcr = calculate_npc(cipher1, cipher2)
    uaci = calculate_uaci(cipher1, cipher2)
    
    return npcr, uaci

# Estadísticas para una imagen original y su cifrado
def compute_statistics(original,ciphered):

    if original.shape != ciphered.shape:
        print("Error: Las imágenes deben tener el mismo tamaño y número de canales.")
        return

    # Estadísticas de la imagen original
    print(f"Estadísticas de la imagen original:")
    ie_original = calculate_ie(original)
    cc_h_o, cc_v_o, cc_d_o = calculate_directional_cc(original)
    chi2_o = chi_square_test(original)

    headers = ["Estadística", "Valor"]
    original_table = [
        ["IE", f"{ie_original:.4f} bits"],
        ["CC-Horiz", f"{cc_h_o:.4f}"],
        ["CC-Vert", f"{cc_v_o:.4f}"],
        ["CC-Diag", f"{cc_d_o:.4f}"],
        ["Chi²", f"{chi2_o:.2f}"]
    ]
    print(tabulate(original_table, headers=headers, tablefmt="pretty"))

    # Estadísticas de la imagen cifrada
    print(f"\nEstadísticas de la imagen cifrada:")
    ie_ciphered = calculate_ie(ciphered)
    cc_h, cc_v, cc_d = calculate_directional_cc(ciphered)
    chi2 = chi_square_test(ciphered)

    ciphered_table = [
        ["IE", f"{ie_ciphered:.4f} bits"],
        ["CC-Horiz", f"{cc_h:.4f}"],
        ["CC-Vert", f"{cc_v:.4f}"],
        ["CC-Diag", f"{cc_d:.4f}"],
        ["Chi²", f"{chi2:.2f}"]
    ]
    print(tabulate(ciphered_table, headers=headers, tablefmt="pretty"))

    # Estadísticas comunes (comparación de las dos imágenes)
    npcr = calculate_npc(original, ciphered)
    uaci = calculate_uaci(original, ciphered)
    cc = calculate_cc(original, ciphered)
    mse = calculate_mse(original, ciphered)
    psnr = calculate_psnr(original, ciphered)

    print("\n[Comunes del Cifrado]")
    common_headers = ["Estadística", "Valor"]
    common_table = [
        ["NPCR", f"{npcr:.4f}%"],
        ["UACI", f"{uaci:.4f}%"],
        ["CC-global", f"{cc:.4f}"],
        ["MSE", f"{mse:.4f}"],
        ["PSNR", f"{psnr:.4f} dB"]
    ]
    print(tabulate(common_table, headers=common_headers, tablefmt="pretty"))


def main():
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Cifrado de imagen y análisis de métricas")
    
    # Argumentos de entrada
    #parser.add_argument("cipher_program_path", help="Ruta al programa de cifrado")
    parser.add_argument("chaos_model", help="Modleo caótico")
    parser.add_argument("input_image", help="Ruta de la imagen a cifrar")
    #parser.add_argument("output_image", help="Ruta donde guardar la imagen cifrada")
    parser.add_argument("password", type=str, help="Contraseña para el cifrado")
    parser.add_argument("rounds", type=int, help="ROndas")
    
    # Otros argumentos adicionales para el cifrado
    parser.add_argument("other_args", nargs=argparse.REMAINDER, help="Otros argumentos para el cifrado")

    # Parseo de los argumentos
    args = parser.parse_args()

    # Verifica si la imagen de entrada existe
    if not os.path.exists(args.input_image):
        print(f"Error: La imagen de entrada '{args.input_image}' no existe.")
        sys.exit(1)

    chaos_model = selectFunction(args.chaos_model)[0]
    original = cv2.imread(args.input_image)

    if np.all(original[:, :, 0] == original[:, :, 1]) and np.all(original[:, :, 1] == original[:, :, 2]):
        original=cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Cifrado de la imagen usando los parámetros proporcionados
    start_time = time.time()
    encrypted_image = encrypt_image(original,chaos_model,args.password, args.rounds)
        #encrypt_image_external_cypher(args.input_image, args.output_image, args.password, args.cipher_program_path, *args.other_args)
    end_time = time.time()
    print(f"Encryption time: {end_time - start_time} s")

    start_time = time.time()
    unencrypted_image = unencrypt_image(encrypted_image,chaos_model,args.password, args.rounds)
    end_time = time.time()
    print(f"Unencryption time: {end_time - start_time} s")
    if encrypted_image is not None:
        
        # Calcula las métricas de la imagen original y cifrada
        plot_histograms(original,encrypted_image)
        compute_statistics(original, encrypted_image)
        
        # También se puede probar la sensibilidad al mensaje
        #npcr, uaci = message_sensitivity_test(original,chaos_model, args.password, args.rounds, *args.other_args)
        #print("\n[Prueba de Sensibilidad al Mensaje]")
        #print(f"\nNPCR: {npcr:.4f}%")
        #print(f"UACI: {uaci:.4f}%")

        npcr, uaci = password_sensitivity_test(original,chaos_model, args.password, args.rounds)
        print("\n[Prueba de Sensibilidad a la contraseña]")
        print(f"\nNPCR: {npcr:.4f}%")
        print(f"UACI: {uaci:.4f}%")


if __name__ == "__main__":
    main()