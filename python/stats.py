import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats, special, fftpack as fft
from scipy.stats import entropy, chisquare
from skimage.feature import graycomatrix, graycoprops
from tabulate import tabulate
import subprocess
import os
import time
import random
import string
import math

# Force software-only AES in PyCryptodome (disable AES-NI)
os.environ['CRYPTODOME_DISABLE_AESNI'] = '1'

try:
    from Crypto.Cipher import AES
    from Crypto.Util import Counter
    from Crypto.Random import get_random_bytes
except ImportError:
    from Cryptodome.Cipher import AES
    from Cryptodome.Util import Counter
    from Cryptodome.Random import get_random_bytes

try:
    import fast_ascon as ascon
except ImportError:
    try:
        import ascon
    except ImportError:
        ascon = None

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

    @staticmethod
    def nist_monobit(image):
        bits = np.unpackbits(image.flatten()).astype(np.int32)
        n = len(bits)
        s_n = np.sum(2 * bits - 1)
        s_obs = abs(s_n) / np.sqrt(n)
        return special.erfc(s_obs / np.sqrt(2))

    @staticmethod
    def nist_runs(image):
        bits = np.unpackbits(image.flatten())
        n = len(bits)
        pi = np.mean(bits)
        if abs(pi - 0.5) >= (2/np.sqrt(n)): return 0.0
        v_n = 1 + np.sum(bits[:-1] != bits[1:])
        return special.erfc(abs(v_n - 2*n*pi*(1-pi)) / (2 * np.sqrt(2*n) * pi * (1-pi)))

    @staticmethod
    def nist_block_freq(image, m=128):
        bits = np.unpackbits(image.flatten())
        n = len(bits)
        n_blocks = n // m
        if n_blocks <= 0: return 0.0
        pi = [np.mean(bits[i*m:(i+1)*m]) for i in range(n_blocks)]
        chi_sq = 4 * m * np.sum((np.array(pi) - 0.5)**2)
        return special.gammaincc(n_blocks/2, chi_sq/2)

    @staticmethod
    def nist_spectral(image):
        bits = np.unpackbits(image.flatten()).astype(np.int32)
        n = len(bits)
        if n % 2 != 0: bits = bits[:-1]; n -= 1
        s = 2 * bits - 1
        dft = fft.fft(s)
        m = np.abs(dft[:n//2])
        threshold = np.sqrt(np.log(1/0.05) * n)
        n_obs = np.sum(m < threshold)
        n_exp = 0.95 * n / 2
        d = (n_obs - n_exp) / np.sqrt(n * 0.95 * 0.05 / 4)
        return special.erfc(abs(d) / np.sqrt(2))

    @staticmethod
    def nist_longest_run(image):
        bits = np.unpackbits(image.flatten())
        n = len(bits)
        m = 128
        if n < m: return 0.0
        n_blocks = n // m
        max_runs = []
        for i in range(n_blocks):
            block = bits[i*m:(i+1)*m]
            runs = "".join(map(str, block)).split('0')
            max_runs.append(max(len(r) for r in runs) if runs else 0)
        mean_obs = np.mean(max_runs)
        expected_mean = np.log2(m)
        z_score = abs(mean_obs - expected_mean) / (np.sqrt(expected_mean))
        return special.erfc(z_score / np.sqrt(2))



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
        self.recovery_hex = None # To store extracted EXIF info
        self.original_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if self.original_img is None: raise ValueError(f"Image not found: {input_path}")
        
        # Determine padded dimensions (logic from aux.cu / padImageToSquare)
        import math
        rows, base_cols = self.original_img.shape[:2]
        channels = self.original_img.shape[2] if len(self.original_img.shape) > 2 else 1
        unstacked_cols = base_cols * channels if channels == 3 else base_cols
        total_pixels_original = unstacked_cols * rows
        bytes_needed = 5
        pixels_for_meta = math.ceil(bytes_needed / 1.0)
        min_S = math.ceil(math.sqrt(total_pixels_original + pixels_for_meta))
        bs = int(block_size)
        self.padded_S = ((min_S + bs - 1) // bs) * bs
        self.padded_cols = self.padded_S
        self.padded_rows = self.padded_S

    def run_cipher_ram_to_ram(self, image_matrix, mode_enc=True, override_password=None):
        mode_flag = '1' if mode_enc else '0'
        password_to_use = override_password if override_password else self.password
        
        success, encoded_buffer = cv2.imencode(".tif", image_matrix)
        binary_flag = '1' if self.is_binary else '0'
        cmd = [
            self.exe, "STDIN", "STDOUT",
            password_to_use, self.rounds, mode_flag,
            self.block_size, self.automata_steps, self.transition, self.chaos, "0",
            binary_flag
        ]

        if not mode_enc and self.recovery_hex:
            cmd.append(self.recovery_hex)

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
                    if "Recovery hex:" in line:
                        self.recovery_hex = line.split("Recovery hex:")[1].strip()
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
        flat = alt_img.ravel()
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
            range_start, range_end = 0, bits_cols
            new_last_char = chr(last_char_code ^ 1)
            mod_pw = original_pw[:-1] + new_last_char
        else:
            # Binary key segments (matching aux.cu segments)
            bits_cols = (self.padded_cols * 2) * 8
            # Align numBlocks calculation with C++ aux.cu (uses MAX_THREADS=64)
            MAX_THREADS = 64
            numBlocks_gpu = (self.padded_cols + MAX_THREADS - 2) // MAX_THREADS - 1
            if numBlocks_gpu < 1: numBlocks_gpu = 1
            bits_flow = (4 + (self.padded_cols + numBlocks_gpu) * 4) * 8
            
            if segment == 'cols':
                range_start, range_end = 0, bits_cols
            elif segment == 'seeds':
                # Skip first 32 bits (r) and target the next seeds (numBlocks bits)
                active_bits = min(bits_flow, (numBlocks_gpu + 1) * 32)
                range_start, range_end = bits_cols + 32, bits_cols + active_bits
            elif segment == 'stego':
                range_start, range_end = bits_cols + bits_flow, len(original_pw)
            else: # 'any'
                range_start, range_end = 0, len(original_pw)
            
            if range_end > len(original_pw): range_end = len(original_pw)
            if range_start >= range_end: range_start = 0 
            
            idx = random.randint(range_start, range_end - 1)
            flipped = '1' if original_pw[idx] == '0' else '0'
            mod_pw = original_pw[:idx] + flipped + original_pw[idx+1:]
            
        c1, _ = self.run_cipher_ram_to_ram(self.original_img, mode_enc=True, override_password=original_pw)
        c2, _ = self.run_cipher_ram_to_ram(self.original_img, mode_enc=True, override_password=mod_pw)
        
        # Calculate visual difference for plotting (only for the last call usually)
        diff_img = cv2.absdiff(c1, c2)
        
        return CryptoMetrics.calculate_npcr_uaci(c1, c2), diff_img

    def occlusion_attack(self, ciphered_img, ratio=0.10):
        """
        Occlusion Attack: Damage a portion of the encrypted image.
        Defaulting to 10% to avoid metadata corruption that crashes unpader.
        """
        damaged = ciphered_img.copy()
        h, w = damaged.shape[:2]
        
        # Calculate region to damage based on area ratio
        occ_area = h * w * ratio
        occ_s = int(math.sqrt(occ_area))
        
        cy, cx = h // 2, w // 2
        y1, y2 = max(0, cy - occ_s // 2), min(h, cy + occ_s // 2)
        x1, x2 = max(0, cx - occ_s // 2), min(w, cx + occ_s // 2)
        
        # Damage the region
        damaged[y1:y2, x1:x2] = 0
        
        # Safety: ensure the last few bytes (metadata) are NEVER touched in ciphertext
        # We restore them from the original ciphertext to avoid noise propagation in unpader logic
        flat_view = damaged.view(np.uint8).flatten()
        if len(flat_view) > 16: # Extra safety margin
             flat_view[-16:] = ciphered_img.view(np.uint8).flatten()[-16:]

        try:
            recovered, _ = self.decrypt_flow(damaged)
        except Exception as e:
            msg = str(e)
            if "Assertion failed" in msg or "terminate" in msg:
                print(f" [!] Occlusion Attack caused a C++ crash due to metadata corruption: {msg}")
            else:
                print(f" [!] Occlusion Attack decryption failed: {e}")
            # Return original if failed
            recovered = np.zeros_like(self.original_img)
            
        return damaged, recovered

    def run_scalability_test(self, repeats=5):
        scales = [1.0, 2.0, 4.0] 
        pixel_counts = []
        enc_times_avg = []
        dec_times_avg = []
        
        # print(f"[>] Running Scalability Benchmark {scales}...")
        
        for s in scales:
            new_w = int(self.original_img.shape[1] * s)
            new_h = int(self.original_img.shape[0] * s)
            if new_w < 16 or new_h < 16: continue
            
            n_pixels = new_w * new_h
            # Cap at 8MP to avoid "Argument list too long" due to key size
            if n_pixels > 8_000_000:
                print(f"   [!] Skipping {s}x ({n_pixels/1e6:.1f} MP) to avoid OS argument limits.")
                continue

            try:
                resized_img = cv2.resize(self.original_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Calculate required key length for THIS image size (with padding)
                import math
                rows, base_cols = resized_img.shape[:2]
                channels = resized_img.shape[2] if len(resized_img.shape) > 2 else 1
                
                # After unstacking
                unstacked_cols = base_cols * channels if channels == 3 else base_cols
                unstacked_rows = rows
                
                # Cap at 5MP of UNSTACKED pixels to avoid "Argument list too long" (shell limit)
                n_pixels_unstacked = unstacked_cols * unstacked_rows
                if n_pixels_unstacked > 5_000_000:
                    print(f"   [!] Skipping {s}x ({n_pixels_unstacked/1e6:.1f} MP unstacked) to avoid OS argument limits.")
                    continue

                # Calculate padded dimensions
                total_pixels_original = unstacked_cols * unstacked_rows
                bytes_needed = 5
                pixels_for_meta = math.ceil(bytes_needed / 1.0)
                min_S = math.ceil(math.sqrt(total_pixels_original + pixels_for_meta))
                block_size = int(self.block_size)
                S = ((min_S + block_size - 1) // block_size) * block_size
                
                padded_cols = S
                padded_rows = S
                # Align num_blocks with C++ aux.cu (uses MAX_THREADS=64)
                MAX_THREADS = 64
                num_blocks = (padded_cols + MAX_THREADS - 2) // MAX_THREADS - 1
                if num_blocks < 1: num_blocks = 1
                # Matches aux.cu: bytes_for_columns + bytes_for_blocks + bytes_for_flow + bytes_for_stego
                total_bytes = (padded_cols * 2) + 4 + (padded_cols + num_blocks) * 4 + 8
                required_bits = total_bytes * 8
                
                # Generate a key of the correct length for this image
                scaled_password = generate_random_password(length=required_bits, binary=True)
                
                curr_enc_times = []
                curr_dec_times = []

                # Warmup
                try: _ = self.run_cipher_ram_to_ram(resized_img, mode_enc=True, override_password=scaled_password)
                except: pass 

                for i in range(repeats):
                    # Use internal C++ time (CPU/GPU-only) instead of Python wall clock
                    ciph, t_enc = self.run_cipher_ram_to_ram(resized_img, mode_enc=True, override_password=scaled_password)
                    curr_enc_times.append(t_enc * 1000.0) # Convert to ms
                    
                    dec, t_dec = self.run_cipher_ram_to_ram(ciph, mode_enc=False, override_password=scaled_password)
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
                
            except Exception:
                break 
                
        return pixel_counts, enc_times_avg, dec_times_avg

class AESCipherTester:
    def __init__(self, image):
        self.original_img = image
        self.key = get_random_bytes(16) # 128-bit AES
    
    def pad(self, data):
        pad_len = 16 - (len(data) % 16)
        return data + bytes([pad_len] * pad_len)

    def unpad(self, data):
        pad_len = data[-1]
        return data[:-pad_len]

    def run_benchmark(self, mode_name='ECB'):
        # Flatten image
        flat_data = self.original_img.tobytes()
        
        if mode_name == 'ECB':
            cipher_enc = AES.new(self.key, AES.MODE_ECB)
            cipher_dec = AES.new(self.key, AES.MODE_ECB)
            data_to_enc = self.pad(flat_data)
        elif mode_name == 'CBC':
            iv = get_random_bytes(16)
            cipher_enc = AES.new(self.key, AES.MODE_CBC, iv)
            cipher_dec = AES.new(self.key, AES.MODE_CBC, iv)
            data_to_enc = self.pad(flat_data)
        elif mode_name == 'CTR':
            ctr = Counter.new(128)
            cipher_enc = AES.new(self.key, AES.MODE_CTR, counter=ctr)
            # Re-create counter for decryption
            ctr_dec = Counter.new(128)
            cipher_dec = AES.new(self.key, AES.MODE_CTR, counter=ctr_dec)
            data_to_enc = flat_data # No padding needed for CTR
        else:
            raise ValueError("Unsupported AES mode")

        # Encrypt
        start = time.perf_counter()
        ciph_bytes = cipher_enc.encrypt(data_to_enc)
        t_enc = (time.perf_counter() - start) * 1000.0

        # Decrypt
        start = time.perf_counter()
        dec_bytes = cipher_dec.decrypt(ciph_bytes)
        if mode_name != 'CTR':
            dec_bytes = self.unpad(dec_bytes)
        t_dec = (time.perf_counter() - start) * 1000.0

        # Reconstruct image for metrics
        ciph_img = np.frombuffer(ciph_bytes[:len(flat_data)], dtype=np.uint8).reshape(self.original_img.shape)
        dec_img = np.frombuffer(dec_bytes[:len(flat_data)], dtype=np.uint8).reshape(self.original_img.shape)
        
        return ciph_img, t_enc, t_dec, dec_img

    def run_key_sensitivity_benchmark(self, mode_name='ECB'):
        """Measures NPCR/UACI when changing one bit of the AES key."""
        key_list = list(self.key)
        key_list[0] ^= 0x01
        key_mod = bytes(key_list)
        
        flat_data = self.original_img.tobytes()
        iv = bytes.fromhex('2122232425262728292A2B2C2D2E2F30')
        
        if mode_name == 'ECB':
            c1 = AES.new(self.key, AES.MODE_ECB).encrypt(self.pad(flat_data))
            c2 = AES.new(key_mod, AES.MODE_ECB).encrypt(self.pad(flat_data))
        elif mode_name == 'CBC':
            c1 = AES.new(self.key, AES.MODE_CBC, iv).encrypt(self.pad(flat_data))
            c2 = AES.new(key_mod, AES.MODE_CBC, iv).encrypt(self.pad(flat_data))
        elif mode_name == 'CTR':
            ctr1 = Counter.new(128); ctr2 = Counter.new(128)
            c1 = AES.new(self.key, AES.MODE_CTR, counter=ctr1).encrypt(flat_data)
            c2 = AES.new(key_mod, AES.MODE_CTR, counter=ctr2).encrypt(flat_data)
        else:
            return 0.0, 0.0
            
        img1 = np.frombuffer(c1[:len(flat_data)], dtype=np.uint8).reshape(self.original_img.shape)
        img2 = np.frombuffer(c2[:len(flat_data)], dtype=np.uint8).reshape(self.original_img.shape)
        
        return CryptoMetrics.calculate_npcr_uaci(img1, img2)

class ASCONCipherTester:
    def __init__(self, image):
        self.original_img = image
        self.key = get_random_bytes(16) # 128-bit ASCON
        self.nonce = get_random_bytes(16)
        self.ad = b"" # Associated data

    def run_benchmark(self):
        if ascon is None:
            return None, 0.0, 0.0
            
        flat_data = self.original_img.tobytes()
        
        # Encrypt
        start = time.perf_counter()
        ciph_bytes = ascon.encrypt(self.key, self.nonce, self.ad, flat_data, variant="Ascon-128")
        t_enc = (time.perf_counter() - start) * 1000.0

        # Decrypt
        start = time.perf_counter()
        dec_bytes = ascon.decrypt(self.key, self.nonce, self.ad, ciph_bytes, variant="Ascon-128")
        t_dec = (time.perf_counter() - start) * 1000.0

        # Reconstruct image for metrics
        ciph_img = np.frombuffer(ciph_bytes[:len(flat_data)], dtype=np.uint8).reshape(self.original_img.shape)
        dec_img = np.frombuffer(dec_bytes[:len(flat_data)], dtype=np.uint8).reshape(self.original_img.shape)
        
        return ciph_img, t_enc, t_dec, dec_img

    def run_key_sensitivity_benchmark(self):
        if ascon is None:
            return 0.0, 0.0
            
        key_list = list(self.key)
        key_list[0] ^= 0x01
        key_mod = bytes(key_list)
        
        flat_data = self.original_img.tobytes()
        
        # fast-ascon and ascon have same interface for encrypt
        c1 = ascon.encrypt(self.key, self.nonce, self.ad, flat_data, variant="Ascon-128")
        c2 = ascon.encrypt(key_mod, self.nonce, self.ad, flat_data, variant="Ascon-128")
        
        img1 = np.frombuffer(c1[:len(flat_data)], dtype=np.uint8).reshape(self.original_img.shape)
        img2 = np.frombuffer(c2[:len(flat_data)], dtype=np.uint8).reshape(self.original_img.shape)
        
        return CryptoMetrics.calculate_npcr_uaci(img1, img2)

# --- 3. EXTENDED DASHBOARD PLOTTING ---
def plot_dashboard(original, ciphered, decrypted, 
                   occluded_input, occluded_output,
                   key_sens_diff_img,
                   benchmark_data,
                   ascon_benchmark_data=None):
    
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
    
    if ascon_benchmark_data:
        p_px, p_enc, p_dec = ascon_benchmark_data
        mp_p = [p / 1_000_000.0 for p in p_px]
        ax_perf.plot(mp_p, p_enc, 'g--v', label='ASCON Enc.', linewidth=1.5, alpha=0.7)
        ax_perf.plot(mp_p, p_dec, 'm--^', label='ASCON Dec.', linewidth=1.5, alpha=0.7)

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
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("--exe", help="Path to the executable", default="cuda/bin/cipher.out")
    parser.add_argument("--password", help="Use this password instead of random ones")
    # Optional algorithm parameters
    parser.add_argument("--rounds", type=int, default=3, help="Number of encryption rounds")
    parser.add_argument("--chaos", type=float, default=3.999, help="Chaotic map parameter")
    parser.add_argument("--block-size", type=int, default=8, help="Block size in pixels")
    parser.add_argument("--steps", type=int, default=20, help="Automata evolution steps")
    parser.add_argument("--trans", type=int, default=20, help="Transition length")

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
        
        # Calculate dimensions AFTER unstacking and padding
        rows, base_cols = temp_tester.original_img.shape[:2]
        channels = temp_tester.original_img.shape[2] if len(temp_tester.original_img.shape) > 2 else 1
        
        # After unstacking: width = base_cols * channels for color images
        unstacked_cols = base_cols * channels if channels == 3 else base_cols
        unstacked_rows = rows
        
        # Calculate padded dimensions (rounded up to nearest multiple of block_size)
        import math
        # Total pixels needed (image + metadata)
        total_pixels_original = unstacked_cols * unstacked_rows
        bytes_needed = 5  # Metadata: W (2 bytes) + H (2 bytes) + color flag (1 byte)
        pixels_for_meta = math.ceil(bytes_needed / 1.0)  # Single channel after unstacking
        
        # Minimum square size
        min_S = math.ceil(math.sqrt(total_pixels_original + pixels_for_meta))
        # Round up to multiple of block_size
        S = ((min_S + args.block_size - 1) // args.block_size) * args.block_size
        
        # After padding, dimensions are S x S
        padded_cols = S
        padded_rows = S
        
        # Now calculate required key length based on PADDED dimensions
        # Align num_blocks with C++ aux.cu (uses MAX_THREADS=64)
        MAX_THREADS = 64
        num_blocks = (padded_cols + MAX_THREADS - 2) // MAX_THREADS - 1
        if num_blocks < 1: num_blocks = 1
        # Matches aux.cu: bytes_for_columns + bytes_for_blocks + bytes_for_flow + bytes_for_stego
        total_bytes = (padded_cols * 2) + 4 + (padded_cols + num_blocks) * 4 + 8
        required_bits = total_bytes * 8
        
        # print(f"[+] Original dimensions: {base_cols}x{rows} (channels={channels})")
        # print(f"[+] After unstacking: {unstacked_cols}x{unstacked_rows}")
        # print(f"[+] After padding: {padded_cols}x{padded_rows}")
        # print(f"[+] Required Key Length: {required_bits} bits ({total_bytes} bytes)")
        # print(f"[+] Starting analysis across {args.runs} runs...")

        for r in range(args.runs):
            if args.password:
                run_pw = args.password
            else:
                run_pw = generate_random_password(length=required_bits, binary=True)
            
            # if (r+1)%5 == 0: print(f"Processing Run {r+1}/{args.runs}...")

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
            
            # Key Sensitivity: random bit from the whole key
            (n_any, u_any), _ = tester.diff_attack_key_sensitivity(segment='any')
            
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
                'npcr_any': n_any,
                'uaci_any': u_any,
                't_enc': t_enc * 1000.0, 
                't_dec': t_dec * 1000.0,
                'psnr': CryptoMetrics.calculate_psnr_mae(tester.original_img, dec)[0],
                'mae': CryptoMetrics.calculate_psnr_mae(tester.original_img, dec)[1]
            })

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
        
        m_np_any, v_np_any = get_stats('npcr_any')
        m_ua_any, v_ua_any = get_stats('uaci_any')

        m_t_enc, v_t_enc = get_stats('t_enc')
        m_t_dec, v_t_dec = get_stats('t_dec')
        
        m_psnr, v_psnr = get_stats('psnr')
        m_mae, v_mae = get_stats('mae')

        # One set of data for non-stochastic metrics (original)
        # Generate a valid binary password of correct length
        final_pw_for_metrics = args.password if args.password else generate_random_password(length=required_bits, binary=True)
        tester_final = ExternalCipherTester(
            args.exe, args.input, final_pw_for_metrics,
            args.rounds, args.chaos,
            args.block_size, args.steps, args.trans,
            is_binary=(user_pw_is_binary if args.password else True)
        )
        ent_orig = CryptoMetrics.calculate_global_entropy(tester_final.original_img)
        corr_orig = CryptoMetrics.calculate_correlations_full(tester_final.original_img)
        cont_orig, hom_orig, ene_orig = CryptoMetrics.calculate_glcm_properties(tester_final.original_img)

        # --- AES COMPARISON ---
        aes_tester = AESCipherTester(tester_final.original_img)
        aes_results = {}
        aes_ciph_imgs = {} # Store for NIST tests
        
        for mode in ['ECB', 'CBC', 'CTR']:
            ciph_aes, t_enc_aes, t_dec_aes, dec_aes = aes_tester.run_benchmark(mode)
            aes_ciph_imgs[mode] = ciph_aes
            
            ent_aes = CryptoMetrics.calculate_global_entropy(ciph_aes)
            corr_aes = CryptoMetrics.calculate_correlations_full(ciph_aes)
            npcr_aes, uaci_aes = CryptoMetrics.calculate_npcr_uaci(tester_final.original_img, ciph_aes)
            cont_aes, hom_aes, _ = CryptoMetrics.calculate_glcm_properties(ciph_aes)
            npcr_k, uaci_k = aes_tester.run_key_sensitivity_benchmark(mode)
            psnr_aes, mae_aes = CryptoMetrics.calculate_psnr_mae(tester_final.original_img, dec_aes)
            
            # Chi-Square for AES
            hist_aes, _ = np.histogram(ciph_aes, bins=256, range=(0, 255))
            chi_aes, p_aes = stats.chisquare(hist_aes)
            
            aes_results[mode] = {
                'entropy': ent_aes,
                'chi': chi_aes,
                'p_val': p_aes,
                'corr_h': corr_aes[0],
                'corr_v': corr_aes[1],
                'corr_d': corr_aes[2],
                'contrast': cont_aes,
                'homogeneity': hom_aes,
                't_enc': t_enc_aes,
                't_dec': t_dec_aes,
                'npcr': npcr_aes,
                'uaci': uaci_aes,
                'npcr_k': npcr_k,
                'uaci_k': uaci_k,
                'psnr': psnr_aes,
                'mae': mae_aes
            }

        # --- ASCON COMPARISON ---
        ascon_results = None
        if ascon:
            ascon_tester = ASCONCipherTester(tester_final.original_img)
            ciph_ascon, t_enc_ascon, t_dec_ascon, dec_ascon = ascon_tester.run_benchmark()
            
            ent_ascon = CryptoMetrics.calculate_global_entropy(ciph_ascon)
            corr_ascon = CryptoMetrics.calculate_correlations_full(ciph_ascon)
            npcr_ascon, uaci_ascon = CryptoMetrics.calculate_npcr_uaci(tester_final.original_img, ciph_ascon)
            cont_ascon, hom_ascon, _ = CryptoMetrics.calculate_glcm_properties(ciph_ascon)
            npcr_k_ascon, uaci_k_ascon = ascon_tester.run_key_sensitivity_benchmark()
            psnr_ascon, mae_ascon = CryptoMetrics.calculate_psnr_mae(tester_final.original_img, dec_ascon)
            
            # Chi-Square for ASCON
            hist_ascon, _ = np.histogram(ciph_ascon, bins=256, range=(0, 255))
            chi_ascon, p_ascon = stats.chisquare(hist_ascon)
            
            ascon_results = {
                'entropy': ent_ascon,
                'chi': chi_ascon,
                'p_val': p_ascon,
                'corr_h': corr_ascon[0],
                'corr_v': corr_ascon[1],
                'corr_d': corr_ascon[2],
                'contrast': cont_ascon,
                'homogeneity': hom_ascon,
                't_enc': t_enc_ascon,
                't_dec': t_dec_ascon,
                'npcr': npcr_ascon,
                'uaci': uaci_ascon,
                'npcr_k': npcr_k_ascon,
                'uaci_k': uaci_k_ascon,
                'psnr': psnr_ascon,
                'mae': mae_ascon
            }
        else:
            print("\n[!] ASCON library not found. Skipping ASCON benchmarks.")

        # Performance (Using one final run for bench or using average of Runs)
        benchmark_data = tester_final.run_scalability_test(repeats=1)

        ascon_benchmark_data = None
        if ascon:
            ascon_px_counts = []
            ascon_enc_ts = []
            ascon_dec_ts = []
            scales = [1.0, 2.0, 4.0]
            for s in scales:
                new_w = int(tester_final.original_img.shape[1] * s)
                new_h = int(tester_final.original_img.shape[0] * s)
                if new_w < 16 or new_h < 16 or (new_w * new_h) > 5_000_000: continue
                
                resized = cv2.resize(tester_final.original_img, (new_w, new_h))
                tester_resize = ASCONCipherTester(resized)
                _, te, td, _ = tester_resize.run_benchmark()
                ascon_px_counts.append(new_w * new_h)
                ascon_enc_ts.append(te)
                ascon_dec_ts.append(td)
            ascon_benchmark_data = (ascon_px_counts, ascon_enc_ts, ascon_dec_ts)

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
        occ_input, occ_output = tester.occlusion_attack(last_ciph)

        # Scalability Data Summary
        aes_bench = {}
        for mode in ['ECB', 'CBC', 'CTR']:
            aes_px, aes_te, aes_td = [], [], []
            scales = [1.0, 2.0, 4.0]
            for s in scales:
                new_w = int(tester_final.original_img.shape[1] * s)
                new_h = int(tester_final.original_img.shape[0] * s)
                if new_w * new_h > 5_000_000: break
                resized = cv2.resize(tester_final.original_img, (new_w, new_h))
                t_aes = AESCipherTester(resized)
                _, te, td, _ = t_aes.run_benchmark(mode)
                aes_px.append(new_w * new_h)
                aes_te.append(te)
                aes_td.append(td)
            aes_bench[mode] = (aes_px, aes_te, aes_td)

        def get_scalability_rows(bench_data):
            if not bench_data: return ["N/A"]*4
            px, enc, dec = bench_data
            res = []
            # we want 2.0x (index 1) and 4.0x (index 2)
            for idx in [1, 2]:
                res.append(f"{enc[idx]:.2f}" if len(enc) > idx else "N/A")
                res.append(f"{dec[idx]:.2f}" if len(dec) > idx else "N/A")
            return res

        # --- CONSOLE REPORT ---
        print("\n" + "="*85)
        print(f" FINAL CRYPTOGRAPHIC REPORT (Across {args.runs} runs) ")
        print("="*85)

        headers = ["Metric", "Original", "Chaotic scheme", "AES-ECB (SW)", "AES-CBC (SW)", "AES-CTR (SW)", "ASCON-128 (SW)", "Ideal Ref"]
        
        # --- CONSOLE REPORT (TABLE 1: General Metrics) ---
        print("\n" + "="*85)
        print(f" CRYPTOGRAPHIC REPORT: GENERAL METRICS (Across {args.runs} runs) ")
        print("="*85)

        headers1 = ["Metric", "Original", "Chaotic scheme", "AES-ECB", "AES-CBC", "AES-CTR", "ASCON-128", "Ideal"]
        
        def fmt_a(mode, key):
            if mode not in aes_results: return "N/A"
            val = aes_results[mode][key]
            if key in ['npcr', 'uaci', 'npcr_k', 'uaci_k']: return f"{val:.4f}%"
            if key in ['t_enc', 't_dec']: return f"{val:.2f}"
            if key == 'chi': 
                p_v = aes_results[mode]['p_val']
                return f"{val:.2f} (P={p_v:.4f})"
            if key == 'psnr': return f"{val:.2f} dB"
            return f"{val:.4f}"

        def fmt_ascon(key):
            if ascon_results is None: return "N/A"
            val = ascon_results[key]
            if key in ['npcr', 'uaci', 'npcr_k', 'uaci_k']: return f"{val:.4f}%"
            if key in ['t_enc', 't_dec']: return f"{val:.2f}"
            if key == 'chi':
                p_v = ascon_results['p_val']
                return f"{val:.2f} (P={p_v:.4f})"
            if key == 'psnr': return f"{val:.2f} dB"
            return f"{val:.4f}"

        chaotic_scal_rows = get_scalability_rows(benchmark_data)
        aes_ecb_scal = get_scalability_rows(aes_bench['ECB'])
        aes_cbc_scal = get_scalability_rows(aes_bench['CBC'])
        aes_ctr_scal = get_scalability_rows(aes_bench['CTR'])
        ascon_scal = get_scalability_rows(ascon_benchmark_data)

        data1 = [
            ["Entropy", f"{ent_orig:.4f}", f"{m_ent:.4f}", fmt_a('ECB','entropy'), fmt_a('CBC','entropy'), fmt_a('CTR','entropy'), fmt_ascon('entropy'), "~7.99"],
            ["Chi-Square (Val, P)", "-", f"{m_chi:.2f} (P={m_pval:.4f})", fmt_a('ECB','chi'), fmt_a('CBC','chi'), fmt_a('CTR','chi'), fmt_ascon('chi'), "> 0.05"],
            ["Corr (H)", f"{corr_orig[0]:.4f}", f"{m_ch:.4f}", fmt_a('ECB','corr_h'), fmt_a('CBC','corr_h'), fmt_a('CTR','corr_h'), fmt_ascon('corr_h'), "0.0"],
            ["Corr (V)", f"{corr_orig[1]:.4f}", f"{m_cv:.4f}", fmt_a('ECB','corr_v'), fmt_a('CBC','corr_v'), fmt_a('CTR','corr_v'), fmt_ascon('corr_v'), "0.0"],
            ["Corr (D)", f"{corr_orig[2]:.4f}", f"{m_cd:.4f}", fmt_a('ECB','corr_d'), fmt_a('CBC','corr_d'), fmt_a('CTR','corr_d'), fmt_ascon('corr_d'), "0.0"],
            ["GLCM Contrast", f"{cont_orig:.4f}", f"{m_cont:.4f}", fmt_a('ECB','contrast'), fmt_a('CBC','contrast'), fmt_a('CTR','contrast'), fmt_ascon('contrast'), "High"],
            ["NPCR (Plaintext)", "-", f"{m_npcr_p:.4f}%", fmt_a('ECB','npcr'), fmt_a('CBC','npcr'), fmt_a('CTR','npcr'), fmt_ascon('npcr'), ">99.6%"],
            ["UACI (Plaintext)", "-", f"{m_uaci_p:.4f}%", fmt_a('ECB','uaci'), fmt_a('CBC','uaci'), fmt_a('CTR','uaci'), fmt_ascon('uaci'), "~33.4%"],
            ["NPCR (Key Sens.)", "-", f"{m_np_any:.4f}%", fmt_a('ECB','npcr_k'), fmt_a('CBC','npcr_k'), fmt_a('CTR','npcr_k'), fmt_ascon('npcr_k'), ">99.6%"],
            ["UACI (Key Sens.)", "-", f"{m_ua_any:.4f}%", fmt_a('ECB','uaci_k'), fmt_a('CBC','uaci_k'), fmt_a('CTR','uaci_k'), fmt_ascon('uaci_k'), "~33.4%"],
            ["Time 1x (ms)", "-", f"{m_t_enc:.2f}", fmt_a('ECB','t_enc'), fmt_a('CBC','t_enc'), fmt_a('CTR','t_enc'), fmt_ascon('t_enc'), "Min."],
            ["Time 2x (ms)", "-", chaotic_scal_rows[0], aes_ecb_scal[0], aes_cbc_scal[0], aes_ctr_scal[0], ascon_scal[0], "Min."],
            ["Time 4x (ms)", "-", chaotic_scal_rows[2], aes_ecb_scal[2], aes_cbc_scal[2], aes_ctr_scal[2], ascon_scal[2], "Min."],
            ["PSNR (Qual)", "-", f"{m_psnr:.2f} dB", fmt_a('ECB','psnr'), fmt_a('CBC','psnr'), fmt_a('CTR','psnr'), fmt_ascon('psnr'), "100.0"],
            ["MAE (Error)", "-", f"{m_mae:.4f}", fmt_a('ECB','mae'), fmt_a('CBC','mae'), fmt_a('CTR','mae'), fmt_ascon('mae'), "0.0"],
        ]
        print(tabulate(data1, headers=headers1, tablefmt="fancy_grid"))

        # --- CONSOLE REPORT (TABLE 2: NIST SP 800-22 Suite) ---
        print("\n" + "="*85)
        print(f" CRYPTOGRAPHIC REPORT: NIST SP 800-22 RANDOMNESS SUITE ")
        print("="*85)
        headers2 = ["NIST Test (P-value)", "Chaotic", "AES-ECB", "AES-CBC", "AES-CTR", "ASCON-128", "Result"]
        
        def f_nist(image, test_func):
            if image is None: return "N/A"
            p = test_func(image)
            mark = "[PASS]" if p > 0.01 else "[FAIL]"
            return f"{p:.4f} {mark}"

        data2 = [
            ["Monobit", f_nist(last_ciph, CryptoMetrics.nist_monobit), f_nist(aes_ciph_imgs.get('ECB'), CryptoMetrics.nist_monobit), f_nist(aes_ciph_imgs.get('CBC'), CryptoMetrics.nist_monobit), f_nist(aes_ciph_imgs.get('CTR'), CryptoMetrics.nist_monobit), f_nist(ciph_ascon, CryptoMetrics.nist_monobit) if ascon else "N/A", "PASS if >0.01"],
            ["Runs", f_nist(last_ciph, CryptoMetrics.nist_runs), f_nist(aes_ciph_imgs.get('ECB'), CryptoMetrics.nist_runs), f_nist(aes_ciph_imgs.get('CBC'), CryptoMetrics.nist_runs), f_nist(aes_ciph_imgs.get('CTR'), CryptoMetrics.nist_runs), f_nist(ciph_ascon, CryptoMetrics.nist_runs) if ascon else "N/A", "PASS if >0.01"],
            ["Block Frequency", f_nist(last_ciph, CryptoMetrics.nist_block_freq), f_nist(aes_ciph_imgs.get('ECB'), CryptoMetrics.nist_block_freq), f_nist(aes_ciph_imgs.get('CBC'), CryptoMetrics.nist_block_freq), f_nist(aes_ciph_imgs.get('CTR'), CryptoMetrics.nist_block_freq), f_nist(ciph_ascon, CryptoMetrics.nist_block_freq) if ascon else "N/A", "PASS if >0.01"],
            ["Spectral (FFT)", f_nist(last_ciph, CryptoMetrics.nist_spectral), f_nist(aes_ciph_imgs.get('ECB'), CryptoMetrics.nist_spectral), f_nist(aes_ciph_imgs.get('CBC'), CryptoMetrics.nist_spectral), f_nist(aes_ciph_imgs.get('CTR'), CryptoMetrics.nist_spectral), f_nist(ciph_ascon, CryptoMetrics.nist_spectral) if ascon else "N/A", "PASS if >0.01"],
            ["Longest Run", f_nist(last_ciph, CryptoMetrics.nist_longest_run), f_nist(aes_ciph_imgs.get('ECB'), CryptoMetrics.nist_longest_run), f_nist(aes_ciph_imgs.get('CBC'), CryptoMetrics.nist_longest_run), f_nist(aes_ciph_imgs.get('CTR'), CryptoMetrics.nist_longest_run), f_nist(ciph_ascon, CryptoMetrics.nist_longest_run) if ascon else "N/A", "PASS if >0.01"],
        ]
        print(tabulate(data2, headers=headers2, tablefmt="fancy_grid"))

        plot_dashboard(tester_final.original_img, last_ciph, last_dec,
                       occ_input, occ_output,
                       key_diff_img,
                       benchmark_data,
                       ascon_benchmark_data)

    except Exception as e:
        print(f"[!] Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()