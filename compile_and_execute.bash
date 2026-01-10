#!/bin/bash
#
# compile_and_execute.bash - Build and test the cipher
#
# This script performs a complete build-encrypt-decrypt cycle to verify
# the cipher works correctly. It's useful for quick testing and validation.
#
# Usage:
#   ./compile_and_execute.bash [rounds]
#
# Arguments:
#   rounds - Number of encryption rounds (default: 3 if not specified)
#
# Example:
#   ./compile_and_execute.bash 3
#
# Process:
#   1. Builds the cipher using 'make' with 8 parallel jobs
#   2. Encrypts a test image (peppers3.tif) with specified rounds
#   3. Decrypts the encrypted image
#   4. Both operations use:
#      - Block size: 8
#      - Automata steps: 50
#      - Transition length: 50
#      - Chaos parameter: 3.9
#
# The encrypted image is saved as: ./cuda/bin/salida.tif
# The decrypted image is saved as: ./cuda/bin/salidaC.tif
#
# Note: You can compare salidaC.tif with the original to verify correctness

# Parse arguments with defaults
ROUNDS=${1:-3}
PRECISION=${2:-float}

# Configure build command
if [ "$PRECISION" == "double" ]; then
    echo "[SCRIPT] Building with Double Precision..."
    BUILD_CMD="make -j 8 PRECISION=double"
else
    echo "[SCRIPT] Building with Standard Float Precision..."
    BUILD_CMD="make -j 8"
fi

# Build the cipher
eval $BUILD_CMD && \
# Encrypt the test image and capture the recovery hex
echo "[SCRIPT] Running encryption..."
ENCRYPT_OUTPUT=$(./cuda/bin/cipher.out ./repositorio/set3/lena3.tif ./cuda/bin/salida.tif password9 $ROUNDS 1 8 20 10 3.9 1 0 2>&1)
echo "$ENCRYPT_OUTPUT"

RECOVERY_HEX=$(echo "$ENCRYPT_OUTPUT" | grep "Recovery hex:" | tail -n 1 | sed 's/.*Recovery hex: \([0-9a-f]*\).*/\1/')
RECOVERY_HEX=$(echo "$RECOVERY_HEX" | tr -d '[:space:]')

if [ -n "$RECOVERY_HEX" ]; then
    echo "[SCRIPT] Captured Recovery Hex: $RECOVERY_HEX"
else
    echo "[WARNING] Could not capture Recovery Hex. Decryption might fail if EXIF reading is broken."
fi

# Decrypt the encrypted image (passing RECOVERY_HEX as 13th argument)
echo -e "\n[SCRIPT] Running decryption..."
./cuda/bin/cipher.out ./cuda/bin/salida.tif ./cuda/bin/salidaC.tif password9 $ROUNDS 0 8 20 10 3.9 0 0 $RECOVERY_HEX