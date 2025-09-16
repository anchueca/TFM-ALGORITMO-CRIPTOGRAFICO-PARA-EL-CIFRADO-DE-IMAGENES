import argparse
import cv2
import numpy as np
import os
from proposed_cipher_cuda import encrypt_image

def main():
    parser = argparse.ArgumentParser(description="Image Encryption Tool")

    parser.add_argument("input_image", help="Path to the image to encrypt")
    parser.add_argument("password", type=str, help="Password for encryption")
    parser.add_argument("--rounds", type=int, default=3, help="Number of encryption rounds")
    parser.add_argument("--show", type=int, default=0, help="Show encryption steps (0-3)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the encrypted image")

    args = parser.parse_args()

    if not os.path.exists(args.input_image):
        print(f"Error: The input image '{args.input_image}' does not exist.")
        exit(1)

    image = cv2.imread(args.input_image)

    encrypted_image = encrypt_image(image, args.password, args.rounds, args.show)

    cv2.imwrite(args.output, encrypted_image)
    print(f"Encrypted image saved at: {args.output}")

if __name__ == "__main__":
    main()
