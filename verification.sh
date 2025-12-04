#!/bin/bash
set -e

INPUT_IMG="/workspaces/TFM-ALGORITMO-CRIPTOGRAFICO-PARA-EL-CIFRADO-DE-IMAGENES/full_report.jpg"
ENCRYPTED_IMG="encrypted.png"
DECRYPTED_IMG="decrypted.png"
PASSWORD="password123"

echo "Encrypting..."
./cuda/bin/cipher.out "$INPUT_IMG" "$ENCRYPTED_IMG" "$PASSWORD" 1 1 16 4 30 10 3.99 1

echo "Decrypting..."
./cuda/bin/cipher.out "$ENCRYPTED_IMG" "$DECRYPTED_IMG" "$PASSWORD" 1 0 16 4 30 10 3.99 1

if [ -f "$DECRYPTED_IMG" ]; then
    echo "Decryption successful, output file created."
else
    echo "Decryption failed, output file not found."
    exit 1
fi
