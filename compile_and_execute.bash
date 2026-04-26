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
        NSYS_CMD=""
    else
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        NSYS_OUT="./cuda/bin/nsys_report_$TIMESTAMP.qdrep"
        NSYS_CMD="nsys profile -o $NSYS_OUT --stats=true"
        echo "[SCRIPT] Profiling enabled. Output will be saved to $NSYS_OUT"
    fi
fi

# Encrypt the test image
echo "[SCRIPT] Running encryption..."
ENCRYPT_CMD="$NSYS_CMD ./cuda/bin/cipher.out ./repositorio/set3/lena3.tif ./cuda/bin/salida.tif password9 $ROUNDS 1 8 20 10 1 0"
ENCRYPT_OUTPUT=$(eval $ENCRYPT_CMD 2>&1)
echo "$ENCRYPT_OUTPUT"

# Extract Recovery Hex
RECOVERY_HEX=$(echo "$ENCRYPT_OUTPUT" | grep "Recovery hex:" | tail -n 1 | sed 's/.*Recovery hex: \([0-9a-f]*\).*/\1/')
RECOVERY_HEX=$(echo "$RECOVERY_HEX" | tr -d '[:space:]')

if [ -n "$RECOVERY_HEX" ]; then
    echo "[SCRIPT] Captured Recovery Hex: $RECOVERY_HEX"
else
    echo "[WARNING] Could not capture Recovery Hex. Decryption might fail if EXIF reading is broken."
fi

# Decrypt the encrypted image
echo -e "\n[SCRIPT] Running decryption..."
DECRYPT_CMD="$NSYS_CMD ./cuda/bin/cipher.out ./cuda/bin/salida.tif ./cuda/bin/salidaC.tif password9 $ROUNDS 0 8 20 10 0 0 $RECOVERY_HEX"
eval $DECRYPT_CMD