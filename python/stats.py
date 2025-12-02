import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import entropy, chisquare
from skimage.feature import graycomatrix, graycoprops
from tabulate import tabulate
import subprocess
import os
import time

# Use 'Agg' backend if headless (no screen), otherwise 'TkAgg'
try:
    matplotlib.use('TkAgg') 
except:
    matplotlib.use('Agg')

# --- 1. MATHEMATICAL METRICS CLASS ---
class CryptoMetrics:
    @staticmethod
    def calculate_global_entropy(image):
        """Calculates Shannon entropy on raw flattened bytes."""
        flat_data = image.flatten()
        hist, _ = np.histogram(flat_data, bins=256, range=[0, 256])
        prob = hist / np.sum(hist)
        return entropy(prob + 1e-14, base=2)

    @staticmethod
    def get_pixel_pairs(image, direction='horizontal', max_samples=3000):
        """Gets pixel pairs (x, y) for correlation, handling color channels."""
        if len(image.shape) == 2:
            channels = [image]
        else:
            channels = cv2.split(image)

        xs_total = []
        ys_total = []

        for chan in channels:
            chan = chan.astype(np.float32)
            if direction == 'horizontal':
                x = chan[:, :-1].flatten()
                y = chan[:, 1:].flatten()
            elif direction == 'vertical':
                x = chan[:-1, :].flatten()
                y = chan[1:, :].flatten()
            elif direction == 'diagonal':
                x = chan[:-1, :-1].flatten()
                y = chan[1:, 1:].flatten()
            else:
                raise ValueError("Invalid direction")
            
            xs_total.append(x)
            ys_total.append(y)

        full_x = np.concatenate(xs_total)
        full_y = np.concatenate(ys_total)

        # Random subsampling for performance
        if len(full_x) > max_samples:
            idx = np.random.choice(len(full_x), max_samples, replace=False)
            return full_x[idx], full_y[idx]
        return full_x, full_y

    @staticmethod
    def calculate_correlations_full(image):
        """Calculates Correlation Coefficients (Horizontal, Vertical, Diagonal)."""
        limit = 500000 
        x, y = CryptoMetrics.get_pixel_pairs(image, 'horizontal', max_samples=limit)
        cc_h = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
        
        x, y = CryptoMetrics.get_pixel_pairs(image, 'vertical', max_samples=limit)
        cc_v = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
        
        x, y = CryptoMetrics.get_pixel_pairs(image, 'diagonal', max_samples=limit)
        cc_d = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
        
        return np.nan_to_num([cc_h, cc_v, cc_d])

    @staticmethod
    def calculate_glcm_properties(image):
        """Calculates GLCM texture metrics: Contrast, Homogeneity, Energy."""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Crop center patch for speed if image is huge
        h, w = gray.shape
        patch_size = 256
        if h > patch_size and w > patch_size:
            patch = gray[h//2 - patch_size//2 : h//2 + patch_size//2, 
                         w//2 - patch_size//2 : w//2 + patch_size//2]
        else:
            patch = gray

        glcm = graycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        
        return contrast, homogeneity, energy

    @staticmethod
    def calculate_npcr_uaci(img1, img2):
        """Calculates differential metrics NPCR and UACI."""
        if img1.shape != img2.shape:
             img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        arr1 = img1.flatten().astype(np.int32)
        arr2 = img2.flatten().astype(np.int32)
        
        # NPCR: Number of Pixel Change Rate
        diff = arr1 != arr2
        npcr = (np.sum(diff) / diff.size) * 100
        
        # UACI: Unified Average Changing Intensity
        abs_diff = np.sum(np.abs(arr1 - arr2))
        uaci = (abs_diff / (diff.size * 255)) * 100
        
        return npcr, uaci

    @staticmethod
    def calculate_chi_square(image):
        """
        Calculates Chi-Square statistic for Histogram Uniformity.
        H0: The distribution is uniform.
        """
        flat_data = image.flatten()
        # Observed frequencies
        hist, _ = np.histogram(flat_data, bins=256, range=[0, 256])
        
        # Expected frequencies (Uniform)
        expected = np.ones(256) * (len(flat_data) / 256)
        
        # Chi-Square Test
        chi2, p_val = chisquare(hist, expected)
        return chi2, p_val

    @staticmethod
    def compute_dft_spectrum(image):
        """Computes 2D Discrete Fourier Transform for frequency analysis."""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        # Magnitude spectrum in Log scale
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
        return magnitude_spectrum

# --- 2. EXECUTION WRAPPER ---
class ExternalCipherTester:
    def __init__(self, exe_path, input_path, password, rounds):
        self.exe = exe_path
        self.input_path = input_path
        self.password = password
        self.rounds = str(rounds)
        self.original_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if self.original_img is None: raise ValueError(f"Image not found: {input_path}")

    def run_cipher_ram_to_ram(self, image_matrix, mode_enc=True, override_password=None):
        mode_flag = '1' if mode_enc else '0'
        password_to_use = override_password if override_password else self.password
        
        success, encoded_buffer = cv2.imencode(".tif", image_matrix)
        if not success: raise ValueError("Python encoding error.")
        
        cmd = [
            self.exe, "STDIN", "STDOUT",
            password_to_use, self.rounds, mode_flag,
            "8", "4", "20", "10", "3.999", "0"
        ]
        try:
            res = subprocess.run(cmd, input=encoded_buffer.tobytes(), capture_output=True, check=True)
            if not res.stdout: raise ValueError("C++ returned empty output")
            nparr = np.frombuffer(res.stdout, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"C++ Error: {e.stderr.decode('utf-8', errors='ignore')}")

    def encrypt_flow(self): return self.run_cipher_ram_to_ram(self.original_img, mode_enc=True)
    def decrypt_flow(self, img): return self.run_cipher_ram_to_ram(img, mode_enc=False)
    
    def diff_attack_plaintext(self):
        """
        Standard Differential Attack: Change 1 bit in PLAINTEXT image.
        """
        alt_img = self.original_img.copy()
        flat = alt_img.view(np.uint8).flatten()
        flat[len(flat)//2] ^= 1 # Flip LSB of center pixel
        
        c1 = self.run_cipher_ram_to_ram(self.original_img, True)
        c2 = self.run_cipher_ram_to_ram(alt_img, True)
        
        if c1 is not None and c2 is not None:
            return CryptoMetrics.calculate_npcr_uaci(c1, c2)
        return (0,0)

    def diff_attack_key_sensitivity(self):
        """
        Key Sensitivity Test: Encrypt SAME image with slightly different PASSWORD.
        """
        # Create modified password (flip last char)
        original_pw = self.password
        if len(original_pw) > 0:
            last_char_code = ord(original_pw[-1])
            new_last_char = chr(last_char_code ^ 1) # Flip 1 bit of last char
            mod_pw = original_pw[:-1] + new_last_char
        else:
            mod_pw = "a" # Fallback
            
        print(f"   [Debug] Key Sensitivity: '{original_pw}' vs '{mod_pw}'")
            
        c1 = self.run_cipher_ram_to_ram(self.original_img, mode_enc=True, override_password=original_pw)
        c2 = self.run_cipher_ram_to_ram(self.original_img, mode_enc=True, override_password=mod_pw)
        
        # Calculate visual difference for plotting
        diff_img = cv2.absdiff(c1, c2)
        
        return CryptoMetrics.calculate_npcr_uaci(c1, c2), diff_img

    def occlusion_attack(self, ciphered_img):
        damaged = ciphered_img.copy()
        h, w = damaged.shape[:2]
        
        # 25% Data Loss
        occ_h = int(h * 0.25)
        occ_w = int(w * 0.25)
        
        cy, cx = h // 2, w // 2
        y1, y2 = cy - occ_h // 2, cy + occ_h // 2
        x1, x2 = cx - occ_w // 2, cx + occ_w // 2
        
        damaged[y1:y2, x1:x2] = 0 
        
        recovered = self.decrypt_flow(damaged)
        return damaged, recovered

    def run_scalability_test(self, repeats=5):
        scales = [0.5, 1.0, 2.0, 4.0] 
        pixel_counts = []
        enc_times_avg = []
        dec_times_avg = []
        
        print(f"[>] Running Scalability Benchmark {scales}...")
        
        for s in scales:
            new_w = int(self.original_img.shape[1] * s)
            new_h = int(self.original_img.shape[0] * s)
            if new_w < 16 or new_h < 16: continue
            
            n_pixels = new_w * new_h
            if n_pixels > 120_000_000:
                print(f"   [!] Skipping {s}x ({n_pixels/1e6:.1f} MP) to prevent crash.")
                continue

            try:
                resized_img = cv2.resize(self.original_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                curr_enc_times = []
                curr_dec_times = []

                # Warmup
                try: _ = self.run_cipher_ram_to_ram(resized_img, mode_enc=True)
                except: pass 

                for i in range(repeats):
                    t0 = time.perf_counter()
                    ciph = self.run_cipher_ram_to_ram(resized_img, mode_enc=True)
                    t1 = time.perf_counter()
                    curr_enc_times.append(t1 - t0)
                    
                    t2 = time.perf_counter()
                    _ = self.run_cipher_ram_to_ram(ciph, mode_enc=False)
                    t3 = time.perf_counter()
                    curr_dec_times.append(t3 - t2)
                
                # Remove outliers
                if repeats >= 3:
                    curr_enc_times.remove(max(curr_enc_times))
                    curr_enc_times.remove(min(curr_enc_times))
                    curr_dec_times.remove(max(curr_dec_times))
                    curr_dec_times.remove(min(curr_dec_times))

                pixel_counts.append(n_pixels)
                enc_times_avg.append(np.mean(curr_enc_times))
                dec_times_avg.append(np.mean(curr_dec_times))
                
                print(f"   -> Scale {s}x ({n_pixels/1e6:.1f} MP): Enc={enc_times_avg[-1]:.4f}s")
                
            except Exception as e:
                print(f"[!] Fail at scale {s}x: {e}")
                break 
                
        return pixel_counts, enc_times_avg, dec_times_avg

# --- 3. EXTENDED DASHBOARD PLOTTING ---
def plot_dashboard(original, ciphered, decrypted, 
                   occluded_input, occluded_output,
                   key_sens_diff_img,
                   benchmark_data):
    
    # Create a 6-Row Dashboard
    fig = plt.figure(figsize=(16, 26)) 
    fig.suptitle("Advanced Cryptographic Analysis & Performance", fontsize=16, fontweight='bold')
    
    def show_img(row, col, img, title, cmap='gray'):
        idx = (row - 1) * 4 + col
        ax = fig.add_subplot(6, 4, idx)
        if img is None: return
        if len(img.shape) == 2: ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        else: ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # --- ROW 1: MAIN IMAGES + KEY SENSITIVITY ---
    show_img(1, 1, original, "Original")
    show_img(1, 2, ciphered, "Encrypted")
    show_img(1, 3, decrypted, "Decrypted")
    # New: Show difference between Cipher(Key1) and Cipher(Key2)
    show_img(1, 4, key_sens_diff_img, "Key Sensitivity Diff (Should be Noise)")

    # --- ROW 2: OCCLUSION & SPECTRUM ---
    show_img(2, 1, occluded_input, "Occluded Input")
    show_img(2, 2, occluded_output, "Occlusion Recovery")
    
    # Frequency Spectrum (DFT)
    ax_fft_orig = fig.add_subplot(6, 4, 7)
    dft_orig = CryptoMetrics.compute_dft_spectrum(original)
    ax_fft_orig.imshow(dft_orig, cmap='inferno')
    ax_fft_orig.set_title("DFT Spectrum (Original)", fontsize=9)
    ax_fft_orig.axis('off')
    
    ax_fft_ciph = fig.add_subplot(6, 4, 8)
    dft_ciph = CryptoMetrics.compute_dft_spectrum(ciphered)
    ax_fft_ciph.imshow(dft_ciph, cmap='inferno')
    ax_fft_ciph.set_title("DFT Spectrum (Encrypted)", fontsize=9)
    ax_fft_ciph.axis('off')

    # --- ROW 3: HISTOGRAMS ---
    for i, (img, col, label) in enumerate(zip([original, ciphered], ['black', 'red'], ['Original', 'Encrypted'])):
        ax = fig.add_subplot(6, 4, 9 + i*2) # Spread them out
        hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
        ax.bar(bins[:-1], hist, color=col, width=1.0)
        ax.set_title(f"{label} Histogram", fontsize=9)
        ax.set_xlim([0, 256])
        ax.get_yaxis().set_visible(False)

    # --- ROW 4: CORRELATIONS (ORIGINAL) ---
    directions = ['horizontal', 'vertical', 'diagonal']
    for i, d in enumerate(directions):
        ax = fig.add_subplot(6, 4, 13 + i)
        x, y = CryptoMetrics.get_pixel_pairs(original, d, max_samples=2000)
        cc = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=0.1, c='black', alpha=0.5)
        ax.set_title(f"Orig {d[:4].capitalize()} (CC: {cc:.4f})", fontsize=8)
        ax.axis('off')

    # --- ROW 5: CORRELATIONS (ENCRYPTED) ---
    for i, d in enumerate(directions):
        ax = fig.add_subplot(6, 4, 17 + i)
        x, y = CryptoMetrics.get_pixel_pairs(ciphered, d, max_samples=2000)
        cc = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=0.1, c='red', alpha=0.5)
        ax.set_title(f"Enc {d[:4].capitalize()} (CC: {cc:.4f})", fontsize=8)
        ax.axis('off')

    # --- ROW 6: PERFORMANCE CHART ---
    ax_perf = plt.subplot2grid((6, 4), (5, 0), colspan=4)
    pixels, t_enc, t_dec = benchmark_data
    
    mp_pixels = [p / 1_000_000.0 for p in pixels] 
    
    ax_perf.plot(mp_pixels, t_enc, 'r-o', label='Encryption', linewidth=2)
    ax_perf.plot(mp_pixels, t_dec, 'b-s', label='Decryption', linewidth=2)
    
    ax_perf.set_title(f"Performance Scalability (0.5x to 10.0x)", fontsize=10)
    ax_perf.set_xlabel("Image Size (Megapixels)")
    ax_perf.set_ylabel("Time (Seconds)")
    ax_perf.legend()
    ax_perf.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.6)
    
    out_file = "full_report.jpg"
    print(f"\n[+] Saving Dashboard to: {out_file}")
    plt.savefig(out_file, dpi=150)

# --- 4. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("password")
    parser.add_argument("exe")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.input): return

    try:
        tester = ExternalCipherTester(args.exe, args.input, args.password, args.rounds)
        
        print("[+] 1. Functional Test (Enc/Dec)...")
        ciphered = tester.encrypt_flow()
        decrypted = tester.decrypt_flow(ciphered)
        if ciphered is None or decrypted is None: return

        print("[+] 2. Analyzing Statistics (Entropy, Correlation, GLCM, Chi-Square)...")
        ent_orig = CryptoMetrics.calculate_global_entropy(tester.original_img)
        ent_ciph = CryptoMetrics.calculate_global_entropy(ciphered)
        corr_orig = CryptoMetrics.calculate_correlations_full(tester.original_img)
        corr_ciph = CryptoMetrics.calculate_correlations_full(ciphered)
        
        # Chi-Square Test
        chi2, p_val = CryptoMetrics.calculate_chi_square(ciphered)
        chi_res = "PASS (Uniform)" if p_val > 0.05 else "FAIL (Reject H0)"
        
        # GLCM Metrics
        cont_orig, hom_orig, ene_orig = CryptoMetrics.calculate_glcm_properties(tester.original_img)
        cont_ciph, hom_ciph, ene_ciph = CryptoMetrics.calculate_glcm_properties(ciphered)
        
        print("[+] 3. Differential Attacks...")
        # Plaintext Sensitivity
        npcr_p, uaci_p = tester.diff_attack_plaintext()
        # Key Sensitivity
        (npcr_k, uaci_k), key_diff_img = tester.diff_attack_key_sensitivity()

        print("[+] 4. Occlusion Attack...")
        occ_input, occ_output = tester.occlusion_attack(ciphered)

        print("[+] 5. Scalability Benchmark (up to 10x)...")
        benchmark_data = tester.run_scalability_test(repeats=1)

        # --- CONSOLE REPORT (ENGLISH) ---
        print("\n" + "="*75)
        print(" FINAL CRYPTOGRAPHIC REPORT ")
        print("="*75)

        headers = ["Metric", "Original", "Encrypted", "Ideal Ref"]
        data = [
            ["Global Entropy",  f"{ent_orig:.5f}",     f"{ent_ciph:.5f}",     "~7.999"],
            ["Chi-Square Test", "-",                   f"{chi2:.2f} (P={p_val:.4f})", "P > 0.05"],
            ["Correlation (H)", f"{corr_orig[0]:.5f}", f"{corr_ciph[0]:.5f}", "0.0"],
            ["Correlation (V)", f"{corr_orig[1]:.5f}", f"{corr_ciph[1]:.5f}", "0.0"],
            ["Correlation (D)", f"{corr_orig[2]:.5f}", f"{corr_ciph[2]:.5f}", "0.0"],
            ["GLCM Contrast",   f"{cont_orig:.2f}",    f"{cont_ciph:.2f}",    "High (>1000)"],
            ["GLCM Homogeneity",f"{hom_orig:.4f}",     f"{hom_ciph:.4f}",     "Low (~0)"],
            ["NPCR (Plaintext)", "-",                  f"{npcr_p:.4f}%",      ">99.6%"],
            ["UACI (Plaintext)", "-",                  f"{uaci_p:.4f}%",      "~33.4%"],
            ["NPCR (Key Sens)",  "-",                  f"{npcr_k:.4f}%",      ">99.6%"],
        ]
        print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

        plot_dashboard(tester.original_img, ciphered, decrypted, 
                       occ_input, occ_output, 
                       key_diff_img,
                       benchmark_data)

    except Exception as e:
        print(f"[!] Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()