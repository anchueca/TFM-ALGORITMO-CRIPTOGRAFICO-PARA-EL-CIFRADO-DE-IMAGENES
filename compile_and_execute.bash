#!/bin/bash
#
# compile_and_execute.bash - Build, test, and optionally profile the cipher
#
# Usage:
#   ./compile_and_execute.bash [rounds] [precision] [profile]
#
# Arguments:
#   rounds    - Number of encryption rounds (default: 3)
#   precision - 'float' or 'double' (default: float)
#   profile   - '1' to profile with Nsight Systems, '0' to skip (default: 0)
#
# Example:
#   ./compile_and_execute.bash 3 float 1

set -o pipefail

# Parse arguments with defaults
ROUNDS=${1:-3}
PRECISION=${2:-float}
PROFILE=${3:-0}   # 0 = no profiling, 1 = use Nsight Systems

# Configure build command
if [ "$PRECISION" == "double" ]; then
    echo "[SCRIPT] Building with Double Precision..."
    BUILD_CMD="make -j 8 PRECISION=double"
else
    echo "[SCRIPT] Building with Standard Float Precision..."
    BUILD_CMD="make -j 8"
fi

# Build the cipher
eval $BUILD_CMD || { echo "[ERROR] Build failed"; exit 1; }

# Prepare Nsight Systems wrapper if profiling
if [ "$PROFILE" -eq 1 ]; then
    if ! command -v nsys &>/dev/null; then
        echo "[WARNING] Nsight Systems not found. Continuing without profiling."
        NSYS_CMD=()
    else
        NSYS_ENABLED=1
    fi
fi

# Process every image while preserving the encryption/decryption flow used by
# the original single-image test.
process_image() {
    local input_path="$1"
    local relative_path="${input_path#./repositorio/}"
    local dataset="${relative_path%%/*}"
    local filename="${relative_path##*/}"
    local stem="${filename%.*}"
    local output_dir="./cuda/bin/results/$dataset"
    local encrypted_path="$output_dir/${stem}.enc.tif"
    local decrypted_path="$output_dir/${stem}.dec.tif"
    local recovery_hex
    local encrypt_output
    local nsys_cmd=()

    mkdir -p "$output_dir"

    if [ "${NSYS_ENABLED:-0}" -eq 1 ]; then
        local timestamp
        local nsys_out
        timestamp=$(date +%Y%m%d_%H%M%S)
        nsys_out="./cuda/bin/nsys_report_${dataset}_${stem}_${timestamp}"
        nsys_cmd=(nsys profile -o "$nsys_out" --stats=true)
        echo "[SCRIPT] Profiling output: $nsys_out"
    fi

    echo -e "\n[SCRIPT] Encrypting: $input_path"
    if ! encrypt_output=$("${nsys_cmd[@]}" ./cuda/bin/cipher.out \
        "$input_path" "$encrypted_path" password9 "$ROUNDS" 1 8 20 10 1 0 2>&1); then
        echo "$encrypt_output"
        echo "[ERROR] Encryption failed for $input_path"
        return 1
    fi
    echo "$encrypt_output"

    recovery_hex=$(echo "$encrypt_output" |
        grep "Recovery hex:" | tail -n 1 |
        sed 's/.*Recovery hex: \([0-9a-fA-F]*\).*/\1/' |
        tr -d '[:space:]')

    if [ -z "$recovery_hex" ]; then
        echo "[ERROR] Could not capture Recovery Hex for $input_path"
        return 1
    fi
    echo "[SCRIPT] Captured Recovery Hex: $recovery_hex"

    echo "[SCRIPT] Decrypting: $encrypted_path"
    if ! "${nsys_cmd[@]}" ./cuda/bin/cipher.out \
        "$encrypted_path" "$decrypted_path" password9 "$ROUNDS" 0 8 20 10 0 0 "$recovery_hex"; then
        echo "[ERROR] Decryption failed for $input_path"
        return 1
    fi
}

mapfile -d '' IMAGES < <(find ./repositorio -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
    -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.bmp' \) \
    -print0 | sort -z)

if [ "${#IMAGES[@]}" -eq 0 ]; then
    echo "[ERROR] No images found under ./repositorio"
    exit 1
fi

FAILED=0
for image_path in "${IMAGES[@]}"; do
    process_image "$image_path" || FAILED=$((FAILED + 1))
done

echo -e "\n[SCRIPT] Processed ${#IMAGES[@]} image(s). Failed: $FAILED"
exit "$FAILED"