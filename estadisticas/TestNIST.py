import struct
import sys
import argparse

from modeloCaos import *

def generate_chaos_data(function, x0, num_values, output_file):
    """
    Generates binarized chaotic data and writes it to a file.

    Args:
        function (callable): Chaotic function.
        x0 (float or list): Initial condition.
        num_values (int): Number of values to generate.
        output_file (str): Output file name.
    """
    bit_buffer = []
    output_bytes = bytearray()

    for n in range(200):
        x0 = function(x0)

    try:
        def write_byte(byte):
            if output_file:
                f.write(struct.pack("B", byte))
            else:
                output_bytes.append(byte)

        f = open(output_file, 'wb')
        print(f"Writing {num_values} values to '{output_file}'...", file=sys.stderr)

        for n in range(num_values):
            x0 = function(x0)
            values = x0 if isinstance(x0, (list, tuple)) else [x0]

            for val in values:
                # Convert float to 64-bit IEEE 754 representation
                float_bits = struct.unpack('>Q', struct.pack('>d', val))[0]

                # Extract 52-bit mantissa
                mantissa = float_bits & ((1 << 52) - 1)

                # Take the top 32 bits of the mantissa
                top32 = mantissa >> (52 - 32)

                # Split into 4 blocks of 8 bits and XOR them
                b0 = (top32 >> 24) & 0xFF
                b1 = (top32 >> 16) & 0xFF
                b2 = (top32 >> 8)  & 0xFF
                b3 = top32 & 0xFF

                result_byte = b0 ^ b1 ^ b2 ^ b3  # Final XOR

                write_byte(result_byte)

            if output_file and num_values > 10 and n % (num_values // 10) == 0:
                print(f"Progress: {(n / num_values) * 100:.2f}%", file=sys.stderr)

        # Write remaining bits if not multiple of 8
        if bit_buffer:
            while len(bit_buffer) < 8:
                bit_buffer.append(0)
            byte = sum((bit << (7 - i)) for i, bit in enumerate(bit_buffer))
            write_byte(byte)

        f.close()
        print("Progress: 100.00%", file=sys.stderr)
        print(f"Writing completed in '{output_file}'.", file=sys.stderr)
        return None

    except Exception as e:
        print(f"An unexpected error occurred during generation: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate values from a chaotic function and optionally analyze them with the NIST STS suite.")
    parser.add_argument("functionName", help="Name of the chaotic function (e.g. logistic, sine, tent)", type=str)
    parser.add_argument("x0", help="Initial condition (floating point number)", type=float)
    parser.add_argument("num_values", help="Number of values to generate", type=int)
    parser.add_argument("--output_file", help="Output file name. If not specified, data is passed to NIST STS.", type=str)
    args = parser.parse_args()

    selected_function = selectFunction(args.functionName)[0]
    if not selected_function:
        print(f"Error: Function '{args.functionName}' not recognized.", file=sys.stderr)
        sys.exit(1)

    initial_state = args.x0

    num_values = args.num_values
    if num_values <= 0:
        print("Error: Number of values to generate must be greater than zero.", file=sys.stderr)
        sys.exit(1)

    return selected_function, initial_state, num_values, args.output_file

if __name__ == "__main__":
    selected_function, initial_state, num_values, output_file = main()
    generate_chaos_data(selected_function, initial_state, num_values, output_file)
