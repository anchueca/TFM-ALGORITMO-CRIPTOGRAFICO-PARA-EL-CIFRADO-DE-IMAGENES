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
# Encrypt the test image
./cuda/bin/cipher.out ./repositorio/set3/peppers3.tif ./cuda/bin/salida.tif password $ROUNDS 1 8 50 50 3.9 1 0 && \
# Decrypt the encrypted image (verbose=0 for cleaner output)
./cuda/bin/cipher.out ./cuda/bin/salida.tif ./cuda/bin/salidaC.tif password $ROUNDS 0 8 50 50 3.9 0 0