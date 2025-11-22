import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import entropy, chisquare
from tabulate import tabulate
import subprocess
import os
import sys

# Use TkAgg for interactive windows, fallback to Agg if headless
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

# --- 1. MATHEMATICAL METRICS CLASS ---
class CryptoMetrics:
    @staticmethod
    def calculate_global_entropy(image):
        """Calculates Shannon entropy of the image."""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        prob = hist / np.sum(hist)
        return entropy(prob.ravel() + 1e-14, base=2)

    @staticmethod
    def get_pixel_pairs(image, direction='horizontal', max_samples=5000):
        """
        Helper to get X, Y pixel arrays for correlation plotting/calculation.
        Directions: 'horizontal', 'vertical', 'diagonal'
        """
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = gray.astype(np.float32)
        
        if direction == 'horizontal':
            x = gray[:, :-1].flatten()
            y = gray[:, 1:].flatten()
        elif direction == 'vertical':
            x = gray[:-1, :].flatten()
            y = gray[1:, :].flatten()
        elif direction == 'diagonal':
            x = gray[:-1, :-1].flatten()
            y = gray[1:, 1:].flatten()
        else:
            raise ValueError("Invalid direction")

        # Subsample to avoid performance issues on huge images
        if len(x) > max_samples:
            idx = np.random.choice(len(x), max_samples, replace=False)
            return x[idx], y[idx]
        return x, y

    @staticmethod
    def calculate_correlations_full(image):
        """Calculates correlation coefficients in all 3 directions."""
        x, y = CryptoMetrics.get_pixel_pairs(image, 'horizontal', max_samples=10**8)
        cc_h = np.corrcoef(x, y)[0, 1]
        x, y = CryptoMetrics.get_pixel_pairs(image, 'vertical', max_samples=10**8)
        cc_v = np.corrcoef(x, y)[0, 1]
        x, y = CryptoMetrics.get_pixel_pairs(image, 'diagonal', max_samples=10**8)
        cc_d = np.corrcoef(x, y)[0, 1]
        return np.nan_to_num([cc_h, cc_v, cc_d])

    @staticmethod
    def calculate_npcr_uaci(img1, img2):
        """Calculates NPCR and UACI between two images."""
        if img1.shape != img2.shape:
             img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        arr1, arr2 = img1.astype(np.int16), img2.astype(np.int16)
        diff = arr1 != arr2
        npcr = (np.sum(diff) / diff.size) * 100
        
        abs_diff = np.sum(np.abs(arr1 - arr2))
        uaci = (abs_diff / (diff.size * 255)) * 100
        return npcr, uaci

    @staticmethod
    def calculate_chi_square(image):
        """Calculates Chi-Square histogram uniformity test."""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        expected = np.ones_like(hist) * np.sum(hist) / 256
        chi2, p_val = chisquare(hist, expected)
        return chi2, p_val

# --- 2. EXECUTION & LOGIC CLASS ---
class ExternalCipherTester:
    def __init__(self, exe_path, input_path, password, rounds):
        self.exe = exe_path
        self.input_path = input_path
        self.password = password
        self.rounds = str(rounds)
        # Load the original image once into memory
        self.original_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if self.original_img is None:
            raise ValueError(f"Could not load input image: {input_path}")

    def run_cipher_ram_to_ram(self, image_matrix, mode_enc=True):
        """
        Full RAM Pipeline:
        Python (Matrix) -> Encode -> Stdin -> C++ -> Stdout -> Python (Matrix)
        """
        if not isinstance(image_matrix, np.ndarray):
             raise ValueError("Input to cipher must be a numpy array (image matrix), not a path.")

        mode_flag = '1' if mode_enc else '0'
        
        # 1. Encode the RAM image to bytes (TIFF format for lossless transfer)
        success, encoded_buffer = cv2.imencode(".tif", image_matrix)
        if not success:
            raise ValueError("Error encoding image in Python before sending to C++")
            
        bytes_to_send = encoded_buffer.tobytes()

        # 2. Configure command
        # Note: First arg is "STDIN", second is "STDOUT"
        cmd = [
            self.exe,
            "STDIN",        # <--- C++ reads from cin
            "STDOUT",       # <--- C++ writes to cout
            self.password,
            self.rounds,
            "0",            # Verbose OFF
            mode_flag,
            "8", "2", "20", "10" # Fixed params
        ]

        try:
            # 3. Execute passing bytes to subprocess input
            res = subprocess.run(
                cmd,
                input=bytes_to_send,  # <--- Inject image bytes here
                capture_output=True,
                check=True
            )
            
            if not res.stdout:
                raise ValueError("C++ returned empty output.")

            # 4. Decode the response
            nparr = np.frombuffer(res.stdout, np.uint8)
            img_decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            return img_decoded

        except subprocess.CalledProcessError as e:
            print(f"[!] C++ Binary Failed.")
            print("Stderr:", e.stderr.decode('utf-8', errors='ignore'))
            raise

    def encrypt_flow(self):
        """Encrypts the original image loaded in memory."""
        return self.run_cipher_ram_to_ram(self.original_img, mode_enc=True)

    def decrypt_flow(self, ciphered_img_ram):
        """Decrypts an image provided in RAM."""
        return self.run_cipher_ram_to_ram(ciphered_img_ram, mode_enc=False)

    def diff_attack(self):
        """
        Performs Differential Attack by flipping 1 bit and comparing results.
        Entirely in memory, no temp files needed.
        """
        # Create a copy to modify
        alt_img = self.original_img.copy()
        
        # Flip LSB safely (works for Gray or Color)
        if len(alt_img.shape) > 2: 
            alt_img[0,0,0] ^= 1
        else: 
            alt_img[0,0] ^= 1
        
        # Encrypt Original (C1)
        # FIX: We pass self.original_img (Matrix), NOT self.input_path (String)
        c1 = self.run_cipher_ram_to_ram(self.original_img, mode_enc=True)
        
        # Encrypt Modified (C2)
        # FIX: We pass alt_img (Matrix)
        c2 = self.run_cipher_ram_to_ram(alt_img, mode_enc=True)
        
        if c1 is not None and c2 is not None:
            return CryptoMetrics.calculate_npcr_uaci(c1, c2)
        
        return (0,0)

# --- 3. COMPLETE VISUALIZATION ---
def plot_results_complete(original, ciphered, decrypted, diff_map):
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Comprehensive Encryption Analysis", fontsize=16, fontweight='bold')

    # ROW 1: IMAGES
    imgs = [original, ciphered, decrypted, diff_map]
    titles = ["Original", "Encrypted", "Decrypted", "Error Map (Black=Perfect)"]
    for i, (img, title) in enumerate(zip(imgs, titles)):
        if i == 3: axs[0, i].imshow(img, cmap='hot', vmin=0, vmax=255)
        else: 
            # Convert BGR to RGB for matplotlib
            if len(img.shape) > 2:
                axs[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                axs[0, i].imshow(img, cmap='gray')
        axs[0, i].set_title(title)
        axs[0, i].axis('off')

    # ROW 2: HISTOGRAMS & HORIZONTAL
    for i, (img, col) in enumerate(zip([original, ciphered], ['black', 'red'])):
        if len(img.shape) > 2: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: gray = img
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        axs[1, i].bar(range(256), hist.ravel(), color=col)
        axs[1, i].set_title(f"Histogram ({'Original' if i==0 else 'Encrypted'})")
        axs[1, i].set_xlim([0, 255])

    # Scatter plots
    directions = ['horizontal', 'horizontal', 'vertical', 'vertical', 'diagonal', 'diagonal']
    sources = [original, ciphered, original, ciphered, original, ciphered]
    colors = ['black', 'red', 'black', 'red', 'black', 'red']
    axes_pos = [(1,2), (1,3), (2,0), (2,1), (2,2), (2,3)]
    
    for (img, direct, col, (r, c)) in zip(sources, directions, colors, axes_pos):
        x, y = CryptoMetrics.get_pixel_pairs(img, direct)
        axs[r, c].scatter(x, y, s=0.5, c=col, alpha=0.5)
        axs[r, c].set_title(f"{direct.capitalize()} ({'Orig' if col=='black' else 'Enc'})")
        axs[r, c].set_xlim(0, 255); axs[r, c].set_ylim(0, 255)
        axs[r, c].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4)

    #plt.show()

    output_filename = "reporte_estadistico.png"
    print(f"\n[+] Saving plot to {output_filename} (Headless Mode)...")
    plt.savefig(output_filename, dpi=150)

# --- 4. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("password")
    parser.add_argument("exe")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: Input file not found: {args.input}")
        return

    try:
        print("[+] Initializing Python Wrapper...")
        tester = ExternalCipherTester(args.exe, args.input, args.password, args.rounds)
        
        # 1. Encrypt (RAM -> RAM)
        print("[+] Encrypting (Full RAM Pipe)...")
        ciphered = tester.encrypt_flow()
        
        # 2. Decrypt (RAM -> RAM)
        print("[+] Decrypting (Full RAM Pipe)...")
        decrypted = tester.decrypt_flow(ciphered)

        if ciphered is None or decrypted is None:
            print("[!] Error: Pipeline failed.")
            return

        # --- STATISTICS ---
        print("[+] Calculating statistics...")
        
        # Integrity
        if tester.original_img.shape != decrypted.shape:
            print("[!] Warning: Shape mismatch. Resizing decrypted for comparison.")
            decrypted = cv2.resize(decrypted, (tester.original_img.shape[1], tester.original_img.shape[0]))

        is_perfect = np.array_equal(tester.original_img, decrypted)
        diff_map = cv2.absdiff(tester.original_img, decrypted)
        diff_pixels = np.count_nonzero(diff_map)
        
        # Entropy
        ent_orig = CryptoMetrics.calculate_global_entropy(tester.original_img)
        ent_ciph = CryptoMetrics.calculate_global_entropy(ciphered)
        
        # Correlation
        corr_orig = CryptoMetrics.calculate_correlations_full(tester.original_img)
        corr_ciph = CryptoMetrics.calculate_correlations_full(ciphered)
        
        # Chi-Square
        chi2, p_val = CryptoMetrics.calculate_chi_square(ciphered)
        
        # Differential Attack
        print("[+] Running Differential Attack...")
        npcr, uaci = tester.diff_attack()

        # --- TABLES ---
        print("\n" + "="*60)
        print(" FULL STATISTICAL REPORT ")
        print("="*60)

        headers_gen = ["Metric", "Original", "Encrypted", "Ideal"]
        table_gen = [
            ["Global Entropy", f"{ent_orig:.4f}", f"{ent_ciph:.4f}", "~7.9990"],
            ["Corr (Horiz)", f"{corr_orig[0]:.4f}", f"{corr_ciph[0]:.4f}", "~0.0000"],
            ["Corr (Vert)",  f"{corr_orig[1]:.4f}", f"{corr_ciph[1]:.4f}", "~0.0000"],
            ["Corr (Diag)",  f"{corr_orig[2]:.4f}", f"{corr_ciph[2]:.4f}", "~0.0000"],
        ]
        print(tabulate(table_gen, headers=headers_gen, tablefmt="fancy_grid"))
        print("")

        chi_res = "Pass (Uniform)" if p_val > 0.05 else "Fail/Suspect"
        table_dist = [
            ["Chi-Square", f"{chi2:.2f}", f"{p_val:.4f}", chi_res]
        ]
        print(tabulate(table_dist, headers=["Test", "Value", "P-Value", "Result"], tablefmt="fancy_grid"))
        print("")

        table_sec = [
            ["NPCR", f"{npcr:.4f} %", "> 99.60 %"],
            ["UACI", f"{uaci:.4f} %", "~ 33.46 %"]
        ]
        print(tabulate(table_sec, headers=["Metric", "Value", "Threshold"], tablefmt="fancy_grid"))
        print("")

        print(f"Integrity Check: {'SUCCESS' if is_perfect else 'FAIL'}")
        if not is_perfect:
            print(f" -> Errors: {diff_pixels} pixels differ.")

        print("\n[+] Launching Plot...")
        plot_results_complete(tester.original_img, ciphered, decrypted, diff_map)

    except Exception as e:
        print(f"\n[!] Python Runtime Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()