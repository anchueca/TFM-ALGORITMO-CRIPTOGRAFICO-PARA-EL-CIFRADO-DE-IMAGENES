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
import random
import string

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
    def calculate_psnr_mae(img1, img2):
        mse = np.mean((img1.astype(float) - img2.astype(float))**2)
        if mse == 0:
            psnr = 100.0  # Lossless
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        mae = np.mean(np.abs(img1.astype(float) - img2.astype(float)))
        return psnr, mae



# --- 2. EXECUTION WRAPPER ---
class ExternalCipherTester:
    def __init__(self, exe_path, input_path, password, rounds, chaos, block_size, automata_steps, transition, is_binary=False):
        self.exe = exe_path
        self.input_path = input_path
        self.password = password
        self.rounds = str(rounds)
        self.chaos = str(chaos)
        self.block_size = str(block_size)
        self.automata_steps = str(automata_steps)
        self.transition = str(transition)
        self.is_binary = is_binary
        self.original_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if self.original_img is None: raise ValueError(f"Image not found: {input_path}")

    def run_cipher_ram_to_ram(self, image_matrix, mode_enc=True, override_password=None):
        mode_flag = '1' if mode_enc else '0'
        password_to_use = override_password if override_password else self.password
        
        success, encoded_buffer = cv2.imencode(".tif", image_matrix)
        if not success: raise ValueError("Python encoding error.")
        
        binary_flag = '1' if self.is_binary else '0'
        
        cmd = [
            self.exe, "STDIN", "STDOUT",
            password_to_use, self.rounds, mode_flag,
            self.block_size, self.automata_steps, self.transition, self.chaos, "0",
            binary_flag
        ]
        try:
            res = subprocess.run(cmd, input=encoded_buffer.tobytes(), capture_output=True, check=True)
            if not res.stdout: raise ValueError("C++ returned empty output")
            
            # Parse EXEC_TIME from stderr
            exec_time = 0.0
            try:
                stderr_str = res.stderr.decode('utf-8', errors='ignore')
                for line in stderr_str.split('\n'):
                    if "EXEC_TIME:" in line:
                        exec_time = float(line.split("EXEC_TIME:")[1].strip())
            except Exception:
                pass

            nparr = np.frombuffer(res.stdout, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED), exec_time
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"C++ Error: {e.stderr.decode('utf-8', errors='ignore')}")

    def encrypt_flow(self): 
        img, t_enc = self.run_cipher_ram_to_ram(self.original_img, mode_enc=True)
        return img, t_enc
    def decrypt_flow(self, img): 
        res, t_dec = self.run_cipher_ram_to_ram(img, mode_enc=False)
        return res, t_dec
    
    def diff_attack_plaintext(self):
        """
        Standard Differential Attack: Change 1 bit in PLAINTEXT image.
        """
        alt_img = self.original_img.copy()
        flat = alt_img.view(np.uint8).flatten()
        flat[len(flat)//2] ^= 1 # Flip LSB of center pixel
        
        c1, _ = self.run_cipher_ram_to_ram(self.original_img, True)
        c2, _ = self.run_cipher_ram_to_ram(alt_img, True)
        
        if c1 is not None and c2 is not None:
            return CryptoMetrics.calculate_npcr_uaci(c1, c2)
        return (0,0)

    def diff_attack_key_sensitivity(self, segment='any'):
        """
        Key Sensitivity Test: Encrypt SAME image with slightly different PASSWORD.
        Can target specific segments: 'rows', 'cols', 'seeds', or 'any'.
        """
        original_pw = self.password
        is_binary = all(c in '01' for c in original_pw) and len(original_pw) > 100
        
        if not is_binary:
            # Fallback for old alphanumeric passwords
            last_char_code = ord(original_pw[-1]) if original_pw else 97
            new_last_char = chr(last_char_code ^ 1)
            mod_pw = original_pw[:-1] + new_last_char
            print(f"   [Debug] Key Sensitivity (Alpha): '{original_pw[:8]}...' vs '{mod_pw[:8]}...'")
        else:
            # Binary key segments
            # 1. Row CA (rows * 2 bytes)
            # 2. Col CA (cols * 2 bytes)
            # 3. Chaotic Seeds (the rest)
            h, w = self.original_img.shape[:2]
            bits_rows = (h * 2) * 8
            bits_cols = (w * 2) * 8
            
            if segment == 'rows':
                range_start, range_end = 0, bits_rows
            elif segment == 'cols':
                range_start, range_end = bits_rows, bits_rows + bits_cols
            elif segment == 'seeds':
                range_start, range_end = bits_rows + bits_cols, len(original_pw)
            else: # 'any'
                range_start, range_end = 0, len(original_pw)
            
            idx = random.randint(range_start, range_end - 1)
            flipped = '1' if original_pw[idx] == '0' else '0'
            mod_pw = original_pw[:idx] + flipped + original_pw[idx+1:]
            print(f"   [Debug] Key Sensitivity ({segment}): Flipped bit {idx} in {segment} segment.")
            
        c1, _ = self.run_cipher_ram_to_ram(self.original_img, mode_enc=True, override_password=original_pw)
        c2, _ = self.run_cipher_ram_to_ram(self.original_img, mode_enc=True, override_password=mod_pw)
        
        # Calculate visual difference for plotting (only for the last call usually)
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
        
        recovered, _ = self.decrypt_flow(damaged)
        return damaged, recovered

    def run_scalability_test(self, repeats=5):
        scales = [1.0, 2.0, 4.0] 
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
                    # Use internal C++ time (CPU/GPU-only) instead of Python wall clock
                    ciph, t_enc = self.run_cipher_ram_to_ram(resized_img, mode_enc=True)
                    curr_enc_times.append(t_enc * 1000.0) # Convert to ms
                    
                    dec, t_dec = self.run_cipher_ram_to_ram(ciph, mode_enc=False)
                    curr_dec_times.append(t_dec * 1000.0) # Convert to ms
                
                # Remove outliers
                if repeats >= 3:
                    curr_enc_times.remove(max(curr_enc_times))
                    curr_enc_times.remove(min(curr_enc_times))
                    curr_dec_times.remove(max(curr_dec_times))
                    curr_dec_times.remove(min(curr_dec_times))

                pixel_counts.append(n_pixels)
                enc_times_avg.append(np.mean(curr_enc_times))
                dec_times_avg.append(np.mean(curr_dec_times))
                
                print(f"   -> Scale {s}x ({n_pixels/1e6:.1f} MP): Enc={enc_times_avg[-1]:.4f} ms")
                
            except Exception as e:
                print(f"[!] Fail at scale {s}x: {e}")
                break 
                
        return pixel_counts, enc_times_avg, dec_times_avg

# --- 3. EXTENDED DASHBOARD PLOTTING ---
def plot_dashboard(original, ciphered, decrypted, 
                   occluded_input, occluded_output,
                   key_sens_diff_img,
                   benchmark_data):
    
    # Create a 5-Row Dashboard (Compact)
    fig = plt.figure(figsize=(16, 20)) 
    fig.suptitle("Advanced Cryptographic Analysis & Performance", fontsize=16, fontweight='bold')
    
    def show_img(row, col, img, title, cmap='gray'):
        idx = (row - 1) * 4 + col
        ax = fig.add_subplot(5, 4, idx)
        if img is None: return
        if len(img.shape) == 2: ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        else: ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # --- ROW 1: MAIN IMAGES + KEY SENSITIVITY ---
    show_img(1, 1, original, "Original")
    show_img(1, 2, ciphered, "Encrypted")
    show_img(1, 3, decrypted, "Decrypted")
    show_img(1, 4, key_sens_diff_img, "Key Sensitivity Diff")

    # --- ROW 2: OCCLUSION & HISTOGRAMS ---
    # Cols 1-2: Occlusion
    show_img(2, 1, occluded_input, "Occluded Input")
    show_img(2, 2, occluded_output, "Occlusion Recovery")
    
    # Cols 3-4: Histograms
    for i, (img, col, label) in enumerate(zip([original, ciphered], ['black', 'red'], ['Original', 'Encrypted'])):
        ax = fig.add_subplot(5, 4, 7 + i) 
        hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
        ax.bar(bins[:-1], hist, color=col, width=1.0)
        ax.set_title(f"{label} Histogram", fontsize=9)
        ax.set_xlim([0, 256])
        ax.get_yaxis().set_visible(False)

    # --- ROW 3: CORRELATIONS (ORIGINAL) ---
    directions = ['horizontal', 'vertical', 'diagonal']
    for i, d in enumerate(directions):
        ax = fig.add_subplot(5, 4, 9 + i)
        x, y = CryptoMetrics.get_pixel_pairs(original, d, max_samples=2000)
        cc = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=0.1, c='black', alpha=0.5)
        ax.set_title(f"Orig {d[:4].capitalize()} (CC: {cc:.4f})", fontsize=8)
        ax.axis('off')

    # --- ROW 4: CORRELATIONS (ENCRYPTED) ---
    for i, d in enumerate(directions):
        ax = fig.add_subplot(5, 4, 13 + i)
        x, y = CryptoMetrics.get_pixel_pairs(ciphered, d, max_samples=2000)
        cc = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=0.1, c='red', alpha=0.5)
        ax.set_title(f"Enc {d[:4].capitalize()} (CC: {cc:.4f})", fontsize=8)
        ax.axis('off')

    # --- ROW 5: PERFORMANCE CHART ---
    ax_perf = plt.subplot2grid((5, 4), (4, 0), colspan=4)
    pixels, t_enc, t_dec = benchmark_data
    
    mp_pixels = [p / 1_000_000.0 for p in pixels] 
    
    ax_perf.plot(mp_pixels, t_enc, 'r-o', label='Encryption', linewidth=2)
    ax_perf.plot(mp_pixels, t_dec, 'b-s', label='Decryption', linewidth=2)
    
    ax_perf.set_title(f"Performance Scalability (0.5x to 10.0x)", fontsize=10)
    ax_perf.set_xlabel("Image Size (Megapixels)")
    ax_perf.set_ylabel("Time (Milliseconds)")
    ax_perf.legend()
    ax_perf.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4)
    

    out_file = "full_report.jpg"
    print(f"\n[+] Saving Dashboard to: {out_file}")
    plt.savefig(out_file, dpi=150)

# --- 4. MAIN ---
def generate_random_password(length=16, binary=False):
    """Generates a random alphanumeric password or a bitstring."""
    if binary:
        return ''.join(random.choices('01', k=length))
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# --- 4. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("exe")
    parser.add_argument("--password", help="Use this password instead of random ones")

    # Optional algorithm parameters
    parser.add_argument("--rounds", type=int, default=3, help="Number of encryption rounds")
    parser.add_argument("--chaos", type=float, default=3.999, help="Chaotic map parameter")
    parser.add_argument("--block-size", type=int, default=8, help="Block size in pixels")
    parser.add_argument("--steps", type=int, default=50, help="Automata evolution steps")
    parser.add_argument("--trans", type=int, default=50, help="Transition length")

    # New analysis parameters
    parser.add_argument("--seed", help="Seed for random password generation")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs with different passwords")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Input file {args.input} not found.")
        return

    # Seed the random generator
    if args.seed is not None:
        random.seed(args.seed)

    try:
        results = []

        # Get image dimensions to calculate required bit length
        # We need a temporary tester to load the image
        temp_tester = ExternalCipherTester(
            args.exe, args.input, "dummy",
            args.rounds, args.chaos, 
            args.block_size, args.steps, args.trans,
            is_binary=False
        )
        # Check if the user-provided password is binary (if any)
        user_pw_is_binary = False
        if args.password:
            user_pw_is_binary = all(c in '01' for c in args.password) and len(args.password) > 100
        rows, base_cols = temp_tester.original_img.shape[:2]
        channels = temp_tester.original_img.shape[2] if len(temp_tester.original_img.shape) > 2 else 1
        cols = base_cols * channels
        num_blocks = (cols + 256) // 256
        total_bytes = (rows * 2) + (cols * 2) + 4 + (cols + num_blocks) * 4
        required_bits = total_bytes * 8
        
        print(f"[+] Required Key Length: {required_bits} bits ({total_bytes} bytes)")
        print(f"[+] Starting analysis across {args.runs} runs...")

        for r in range(args.runs):
            if args.password:
                run_pw = args.password
            else:
                run_pw = generate_random_password(length=required_bits, binary=True)
            
            print(f"\n[>] Run {r+1}/{args.runs} | Key: {run_pw[:16]}...{run_pw[-16:]} ({len(run_pw)} bits)")

            tester = ExternalCipherTester(
                args.exe, args.input, run_pw,
                args.rounds, args.chaos,
                args.block_size, args.steps, args.trans,
                is_binary=(user_pw_is_binary if args.password else True)
            )

            # Encrypt/Decrypt
            ciph, t_enc = tester.encrypt_flow()
            dec, t_dec = tester.decrypt_flow(ciph)

            if ciph is None or dec is None:
                print(f" [!] Run {r+1} failed.")
                continue

            # Core Metrics
            ent_orig = CryptoMetrics.calculate_global_entropy(tester.original_img)
            ent_ciph = CryptoMetrics.calculate_global_entropy(ciph)
            corr_ciph = CryptoMetrics.calculate_correlations_full(ciph)
            chi2, p_val = CryptoMetrics.calculate_chi_square(ciph)
            cont, hom, ene = CryptoMetrics.calculate_glcm_properties(ciph)

            # Sensitivity Test (Differential)
            npcr_p, uaci_p = tester.diff_attack_plaintext()
            
            # Key Sensitivity per Segment
            (n_rows, u_rows), _ = tester.diff_attack_key_sensitivity(segment='rows')
            (n_cols, u_cols), _ = tester.diff_attack_key_sensitivity(segment='cols')
            (n_seeds, u_seeds), _ = tester.diff_attack_key_sensitivity(segment='seeds')
            
            results.append({
                'entropy': ent_ciph,
                'corr_h': corr_ciph[0],
                'corr_v': corr_ciph[1],
                'corr_d': corr_ciph[2],
                'chi2': chi2,
                'p_val': p_val,
                'glcm_contrast': cont,
                'glcm_homogeneity': hom,
                'glcm_energy': ene,
                'npcr_p': npcr_p,
                'uaci_p': uaci_p,
                'npcr_rows': n_rows,
                'uaci_rows': u_rows,
                'npcr_cols': n_cols,
                'uaci_cols': u_cols,
                'npcr_seeds': n_seeds,
                'uaci_seeds': u_seeds,
                't_enc': t_enc * 1000.0, 
                't_dec': t_dec * 1000.0,
                'psnr': CryptoMetrics.calculate_psnr_mae(tester.original_img, dec)[0],
                'mae': CryptoMetrics.calculate_psnr_mae(tester.original_img, dec)[1]
            })

            # For visual dashboard, keep the last results
            if r == args.runs - 1:
                last_original = tester.original_img
                last_ciph = ciph
                last_dec = dec

        if not results:
            print("[!] No results collected.")
            return

        # --- STATISTICAL ANALYSIS ---
        def get_stats(key):
            vals = [res[key] for res in results]
            return np.mean(vals), np.var(vals)

        m_ent, v_ent = get_stats('entropy')
        m_ch, v_ch = get_stats('corr_h')
        m_cv, v_cv = get_stats('corr_v')
        m_cd, v_cd = get_stats('corr_d')
        m_chi, v_chi = get_stats('chi2')
        m_pval, v_pval = get_stats('p_val')
        m_cont, v_cont = get_stats('glcm_contrast')
        m_hom, v_hom = get_stats('glcm_homogeneity')
        m_npcr_p, v_npcr_p = get_stats('npcr_p')
        m_uaci_p, v_uaci_p = get_stats('uaci_p')
        
        m_np_rows, v_np_rows = get_stats('npcr_rows')
        m_ua_rows, v_ua_rows = get_stats('uaci_rows')
        m_np_cols, v_np_cols = get_stats('npcr_cols')
        m_ua_cols, v_ua_cols = get_stats('uaci_cols')
        m_np_seeds, v_np_seeds = get_stats('npcr_seeds')
        m_ua_seeds, v_ua_seeds = get_stats('uaci_seeds')

        m_t_enc, v_t_enc = get_stats('t_enc')
        m_t_dec, v_t_dec = get_stats('t_dec')
        
        m_psnr, v_psnr = get_stats('psnr')
        m_mae, v_mae = get_stats('mae')

        # One set of data for non-stochastic metrics (original)
        # Use a dummy password as it's not used for original image metrics
        tester_final = ExternalCipherTester(
            args.exe, args.input, "dummy",
            args.rounds, args.chaos,
            args.block_size, args.steps, args.trans,
            is_binary=False
        )
        ent_orig = CryptoMetrics.calculate_global_entropy(tester_final.original_img)
        corr_orig = CryptoMetrics.calculate_correlations_full(tester_final.original_img)
        cont_orig, hom_orig, ene_orig = CryptoMetrics.calculate_glcm_properties(tester_final.original_img)

        # Performance (Using one final run for bench or using average of Runs)
        print("\n[+] Running Performance Scalability Test (1x Repeats)...")
        benchmark_data = tester_final.run_scalability_test(repeats=1)

        # Key Sensitivity Diff Image (One final sample)
        # Use a new random binary password of correct length
        final_pw_for_key_sens = args.password if args.password else generate_random_password(length=required_bits, binary=True)
        tester_final_key_sens = ExternalCipherTester(
            args.exe, args.input, final_pw_for_key_sens,
            args.rounds, args.chaos,
            args.block_size, args.steps, args.trans,
            is_binary=(user_pw_is_binary if args.password else True)
        )
        (npcr_k_sample, uaci_k_sample), key_diff_img = tester_final_key_sens.diff_attack_key_sensitivity()

        # Occlusion Attack (One final sample)
        # Requires a ciphered image, use the last one from the runs
        occ_input, occ_output = tester_final.occlusion_attack(last_ciph)

        # --- CONSOLE REPORT ---
        print("\n" + "="*85)
        print(f" FINAL CRYPTOGRAPHIC REPORT (Across {args.runs} runs) ")
        print("="*85)

        headers = ["Metric", "Original", "Mean", "Variance", "Ideal Ref"]
        data = [
            ["Global Entropy",  f"{ent_orig:.4f}",     f"{m_ent:.4f}",  f"{v_ent:.4f}", "~7.999"],
            ["Chi-Square Test", "-",                   f"{m_chi:.4f} (P={m_pval:.4f})",  f"{v_chi:.4f} (P={v_pval:.4f})", "P > 0.05"],
            ["Correlation (H)", f"{corr_orig[0]:.4f}", f"{m_ch:.4f}",   f"{v_ch:.4f}",  "0.0"],
            ["Correlation (V)", f"{corr_orig[1]:.4f}", f"{m_cv:.4f}",   f"{v_cv:.4f}",  "0.0"],
            ["Correlation (D)", f"{corr_orig[2]:.4f}", f"{m_cd:.4f}",   f"{v_cd:.4f}",  "0.0"],
            ["GLCM Contrast",   f"{cont_orig:.4f}",    f"{m_cont:.4f}", f"{v_cont:.4f}", "High"],
            ["GLCM Homogeneity",f"{hom_orig:.4f}",     f"{m_hom:.4f}",  f"{v_hom:.4f}", "~0"],
            ["NPCR (Plaintext)", "-",                  f"{m_npcr_p:.4f}%", f"{v_npcr_p:.4f}", ">99.6%"],
            ["UACI (Plaintext)", "-",                  f"{m_uaci_p:.4f}%", f"{v_uaci_p:.4f}", "~33.4%"],
            ["NPCR (Key: Rows)", "-",                  f"{m_np_rows:.4f}%", f"{v_np_rows:.4f}", ">99.6%"],
            ["UACI (Key: Rows)", "-",                  f"{m_ua_rows:.4f}%", f"{v_ua_rows:.4f}", "~33.4%"],
            ["NPCR (Key: Cols)", "-",                  f"{m_np_cols:.4f}%", f"{v_np_cols:.4f}", ">99.6%"],
            ["UACI (Key: Cols)", "-",                  f"{m_ua_cols:.4f}%", f"{v_ua_cols:.4f}", "~33.4%"],
            ["NPCR (Key: Seeds)", "-",                 f"{m_np_seeds:.4f}%", f"{v_np_seeds:.4f}", ">99.6%"],
            ["UACI (Key: Seeds)", "-",                 f"{m_ua_seeds:.4f}%", f"{v_ua_seeds:.4f}", "~33.4%"],
            ["Enc. Time (ms)",    "-",                 f"{m_t_enc:.4f}", f"{v_t_enc:.4f}", "Min."],
            ["Dec. Time (ms)",    "-",                 f"{m_t_dec:.4f}", f"{v_t_dec:.4f}", "Min."],
            ["PSNR (Dec. Quality)", "-",               f"{m_psnr:.4f} dB", f"{v_psnr:.4f}", ">50dB"],
            ["MAE (Dec. Error)",   "-",                f"{m_mae:.4f}", f"{v_mae:.4f}", "0.0"],
        ]
        print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

        plot_dashboard(tester_final.original_img, last_ciph, last_dec,
                       occ_input, occ_output,
                       key_diff_img,
                       benchmark_data)

    except Exception as e:
        print(f"[!] Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()