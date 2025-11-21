import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import entropy, chisquare
from tabulate import tabulate
import subprocess
import os
import sys

# --- 1. MATHEMATICAL METRICS CLASS ---
class CryptoMetrics:
    @staticmethod
    def calculate_global_entropy(image):
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

        # Subsample for plotting performance if needed
        if len(x) > max_samples:
            idx = np.random.choice(len(x), max_samples, replace=False)
            return x[idx], y[idx]
        return x, y

    @staticmethod
    def calculate_correlations_full(image):
        # We use the full dataset for calculation (not subsampled)
        # Horizontal
        x, y = CryptoMetrics.get_pixel_pairs(image, 'horizontal', max_samples=10**8)
        cc_h = np.corrcoef(x, y)[0, 1]

        # Vertical
        x, y = CryptoMetrics.get_pixel_pairs(image, 'vertical', max_samples=10**8)
        cc_v = np.corrcoef(x, y)[0, 1]

        # Diagonal
        x, y = CryptoMetrics.get_pixel_pairs(image, 'diagonal', max_samples=10**8)
        cc_d = np.corrcoef(x, y)[0, 1]
        
        return np.nan_to_num([cc_h, cc_v, cc_d])

    @staticmethod
    def calculate_npcr_uaci(img1, img2):
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
        self.input = input_path
        self.password = password
        self.rounds = str(rounds)
        self.original_img = cv2.imread(input_path)
        if self.original_img is None:
            raise ValueError(f"Could not load input image: {input_path}")

    def run_cmd(self, args):
        try:
            subprocess.run([self.exe] + args, check=True, capture_output=True, timeout=60)
        except Exception as e:
            print(f"[!] Error executing C++ binary: {e}")
            raise 

    def encrypt(self, in_file, out_file):
        self.run_cmd([in_file, out_file, self.password, self.rounds, '0', '1', '8', '2', '20', '10'])
        return cv2.imread(out_file)

    def decrypt(self, in_file, out_file):
        self.run_cmd([in_file, out_file, self.password, self.rounds, '0', '0', '8', '2', '20', '10'])
        return cv2.imread(out_file)

    def diff_attack(self):
        alt_img = self.original_img.copy()
        # Flip LSB safely
        if len(alt_img.shape) > 2: alt_img[0,0,0] ^= 1
        else: alt_img[0,0] ^= 1
        
        f_orig, f_alt = "temp_atk_orig.tif", "temp_atk_alt.tif"
        f_c1, f_c2 = "temp_atk_c1.tif", "temp_atk_c2.tif"
        files = [f_orig, f_alt, f_c1, f_c2]
        
        try:
            cv2.imwrite(f_orig, self.original_img)
            cv2.imwrite(f_alt, alt_img)
            c1 = self.encrypt(f_orig, f_c1)
            c2 = self.encrypt(f_alt, f_c2)
            if c1 is not None and c2 is not None:
                return CryptoMetrics.calculate_npcr_uaci(c1, c2)
            return (0,0)
        finally:
            for f in files:
                if os.path.exists(f): os.remove(f)

# --- 3. COMPLETE VISUALIZATION ---
def plot_results_complete(original, ciphered, decrypted, diff_map):
    # Create a 3x4 Grid
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Comprehensive Encryption Analysis", fontsize=16, fontweight='bold')

    # --- ROW 1: IMAGES ---
    imgs = [original, ciphered, decrypted, diff_map]
    titles = ["Original", "Encrypted", "Decrypted", "Error Map (Black=Perfect)"]
    for i, (img, title) in enumerate(zip(imgs, titles)):
        if i == 3: axs[0, i].imshow(img, cmap='hot', vmin=0, vmax=255)
        else: axs[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0, i].set_title(title)
        axs[0, i].axis('off')

    # --- ROW 2: HISTOGRAMS & HORIZONTAL CORRELATION ---
    
    # Histograms (Cols 0 & 1)
    for i, (img, col) in enumerate(zip([original, ciphered], ['black', 'red'])):
        if len(img.shape) > 2: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: gray = img
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        axs[1, i].bar(range(256), hist.ravel(), color=col)
        axs[1, i].set_title(f"Histogram ({'Original' if i==0 else 'Encrypted'})")
        axs[1, i].set_xlim([0, 255])

    # Horizontal Scatter (Cols 2 & 3)
    x_o, y_o = CryptoMetrics.get_pixel_pairs(original, 'horizontal')
    axs[1, 2].scatter(x_o, y_o, s=0.5, c='black', alpha=0.5)
    axs[1, 2].set_title("Horizontal Corr. (Original)")
    axs[1, 2].set_xlabel("Pixel(i, j)"); axs[1, 2].set_ylabel("Pixel(i, j+1)")

    x_c, y_c = CryptoMetrics.get_pixel_pairs(ciphered, 'horizontal')
    axs[1, 3].scatter(x_c, y_c, s=0.5, c='red', alpha=0.5)
    axs[1, 3].set_title("Horizontal Corr. (Encrypted)")
    axs[1, 3].set_xlabel("Pixel(i, j)"); axs[1, 3].set_ylabel("Pixel(i, j+1)")

    # --- ROW 3: VERTICAL & DIAGONAL CORRELATION ---

    # Vertical (Cols 0 & 1)
    x_ov, y_ov = CryptoMetrics.get_pixel_pairs(original, 'vertical')
    axs[2, 0].scatter(x_ov, y_ov, s=0.5, c='black', alpha=0.5)
    axs[2, 0].set_title("Vertical Corr. (Original)")
    axs[2, 0].set_xlabel("Pixel(i, j)"); axs[2, 0].set_ylabel("Pixel(i+1, j)")

    x_cv, y_cv = CryptoMetrics.get_pixel_pairs(ciphered, 'vertical')
    axs[2, 1].scatter(x_cv, y_cv, s=0.5, c='red', alpha=0.5)
    axs[2, 1].set_title("Vertical Corr. (Encrypted)")
    axs[2, 1].set_xlabel("Pixel(i, j)"); axs[2, 1].set_ylabel("Pixel(i+1, j)")

    # Diagonal (Cols 2 & 3)
    x_od, y_od = CryptoMetrics.get_pixel_pairs(original, 'diagonal')
    axs[2, 2].scatter(x_od, y_od, s=0.5, c='black', alpha=0.5)
    axs[2, 2].set_title("Diagonal Corr. (Original)")
    axs[2, 2].set_xlabel("Pixel(i, j)"); axs[2, 2].set_ylabel("Pixel(i+1, j+1)")

    x_cd, y_cd = CryptoMetrics.get_pixel_pairs(ciphered, 'diagonal')
    axs[2, 3].scatter(x_cd, y_cd, s=0.5, c='red', alpha=0.5)
    axs[2, 3].set_title("Diagonal Corr. (Encrypted)")
    axs[2, 3].set_xlabel("Pixel(i, j)"); axs[2, 3].set_ylabel("Pixel(i+1, j+1)")

    # Formatting axis for correlations
    for ax in axs.flatten()[6:]: # Apply to all scatter plots
        ax.set_xlim(0, 255); ax.set_ylim(0, 255)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4)
    plt.show()

# --- 4. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("password")
    parser.add_argument("exe")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    f_cipher = "temp_main_cipher.tif"
    f_decipher = "temp_main_decipher.tif"

    try:
        print("[+] Initializing...")
        tester = ExternalCipherTester(args.exe, args.input, args.password, args.rounds)
        
        print("[+] Encrypting...")
        ciphered = tester.encrypt(args.input, f_cipher)
        
        print("[+] Decrypting...")
        decrypted = tester.decrypt(f_cipher, f_decipher)

        if ciphered is None or decrypted is None:
            print("[!] Error: Failed to generate images.")
            return

        # --- STATISTICS ---
        print("[+] Calculating all statistics...")
        
        # 1. Integrity
        is_perfect = np.array_equal(tester.original_img, decrypted)
        diff_map = cv2.absdiff(tester.original_img, decrypted)
        diff_pixels = np.count_nonzero(diff_map)
        
        # 2. Entropy
        ent_orig = CryptoMetrics.calculate_global_entropy(tester.original_img)
        ent_ciph = CryptoMetrics.calculate_global_entropy(ciphered)
        
        # 3. Correlation (All 3 directions)
        corr_orig = CryptoMetrics.calculate_correlations_full(tester.original_img)
        corr_ciph = CryptoMetrics.calculate_correlations_full(ciphered)
        
        # 4. Chi-Square
        chi2, p_val = CryptoMetrics.calculate_chi_square(ciphered)
        
        # 5. Differential Attack
        npcr, uaci = tester.diff_attack()

        # --- TABLES ---
        print("\n" + "="*60)
        print(" FULL STATISTICAL REPORT ")
        print("="*60)

        # Table 1: General Metrics
        headers_gen = ["Metric", "Original Image", "Encrypted Image", "Ideal (Ref)"]
        table_gen = [
            ["Global Entropy", f"{ent_orig:.4f}", f"{ent_ciph:.4f}", "~7.9990"],
            ["Correlation (Horiz)", f"{corr_orig[0]:.4f}", f"{corr_ciph[0]:.4f}", "~0.0000"],
            ["Correlation (Vert)",  f"{corr_orig[1]:.4f}", f"{corr_ciph[1]:.4f}", "~0.0000"],
            ["Correlation (Diag)",  f"{corr_orig[2]:.4f}", f"{corr_ciph[2]:.4f}", "~0.0000"],
        ]
        print(tabulate(table_gen, headers=headers_gen, tablefmt="fancy_grid"))
        print("")

        # Table 2: Randomness & Distribution
        headers_dist = ["Test", "Value", "P-Value", "Result Interpretation"]
        chi_res = "Passed (Uniform)" if p_val > 0.05 else "Suspect (Not Uniform)" # Simplified interpretation
        if chi2 > 300: chi_res = "High Deviation"
        
        table_dist = [
            ["Chi-Square Statistic", f"{chi2:.2f}", f"{p_val:.4f}", chi_res]
        ]
        print(tabulate(table_dist, headers=headers_dist, tablefmt="fancy_grid"))
        print("")

        # Table 3: Resistance to Attacks
        headers_sec = ["Differential Attack", "Obtained", "Ideal Threshold"]
        table_sec = [
            ["NPCR (Pixel Change Rate)", f"{npcr:.4f} %", "> 99.60 %"],
            ["UACI (Avg Intensity Change)", f"{uaci:.4f} %", "~ 33.46 %"]
        ]
        print(tabulate(table_sec, headers=headers_sec, tablefmt="fancy_grid"))
        print("")

        # Table 4: Integrity
        print(f"Integrity Check: {'SUCCESS' if is_perfect else 'FAIL'}")
        if not is_perfect:
            print(f" -> Errors found: {diff_pixels} pixels differ.")

        print("\n[+] Displaying comprehensive plots...")
        plot_results_complete(tester.original_img, ciphered, decrypted, diff_map)

    except Exception as e:
        print(f"\n[!] Runtime Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        for f in [f_cipher, f_decipher]:
            if os.path.exists(f): 
                try: os.remove(f) 
                except: pass

if __name__ == "__main__":
    main()