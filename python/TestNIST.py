import struct
import sys
import argparse
import Chaos_Generator  # Imports the file created above

def generate_chaos_data(function, x0, num_values, output_file):
    """
    Generates binarized chaotic data and writes it to a file using buffered I/O.

    Args:
        function (callable): Chaotic function.
        x0 (float or list): Initial condition.
        num_values (int): Number of values to generate.
        output_file (str): Output file name.
    """
    
    # Configuration
    BUFFER_SIZE = 65536  # Write to disk in 64KB chunks for performance
    byte_buffer = bytearray()
    
    # Burn-in period (transient removal)
    # We ignore these values to let the system settle onto the attractor
    for _ in range(200):
        x0 = function(x0)

    try:
        # Open file in binary write mode using a context manager
        # If output_file is None, we technically don't write to disk in this snippet,
        # but for this logic we assume a file path is provided or handle sys.stdout.buffer
        
        fd = open(output_file, 'wb') if output_file else None
        
        print(f"Generating {num_values} values...", file=sys.stderr)

        for n in range(num_values):
            x0 = function(x0)
            
            # Handle both 1D (float) and nD (list/tuple) maps
            values = x0 if isinstance(x0, (list, tuple)) else [x0]

            for val in values:
                # 1. Convert float to 64-bit IEEE 754 representation (double)
                # 'd' is double, 'Q' is unsigned long long (8 bytes)
                float_bits = struct.unpack('>Q', struct.pack('>d', val))[0]

                # 2. Extract 52-bit mantissa
                # The mantissa contains the chaotic "fractional" part which is most sensitive
                mantissa = float_bits & ((1 << 52) - 1)

                # 3. Take the top 32 bits of the mantissa
                top32 = mantissa >> (52 - 32)

                # 4. Whitening: Split into 4 blocks of 8 bits and XOR them
                # This helps reduce bias in the resulting byte
                b0 = (top32 >> 24) & 0xFF
                b1 = (top32 >> 16) & 0xFF
                b2 = (top32 >> 8)  & 0xFF
                b3 = top32 & 0xFF

                result_byte = b0 ^ b1 ^ b2 ^ b3 

                # Add to buffer
                byte_buffer.append(result_byte)

            # Flush buffer to file if it is full
            if fd and len(byte_buffer) >= BUFFER_SIZE:
                fd.write(byte_buffer)
                byte_buffer.clear()

            # Progress bar
            if num_values > 100 and n % (num_values // 10) == 0:
                print(f"Progress: {(n / num_values) * 100:.1f}%", file=sys.stderr)

        # Write remaining data in buffer
        if fd and len(byte_buffer) > 0:
            fd.write(byte_buffer)

        print("Progress: 100.00%", file=sys.stderr)
        
        if fd:
            fd.close()
            print(f"Writing completed in '{output_file}'.", file=sys.stderr)

    except IOError as e:
        print(f"File I/O Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate chaotic binary data.")
    parser.add_argument("functionName", help="Name of the chaotic function (logistic, sine, tent, henon)", type=str)
    parser.add_argument("x0", help="Initial condition (float). For Henon, x=x0, y=0", type=float)
    parser.add_argument("num_values", help="Number of iterations to generate", type=int)
    parser.add_argument("--output_file", help="Output file name", type=str, required=True)
    
    args = parser.parse_args()

    # Select function from the module
    selected_function, is_multidimensional = Chaos_Generator.selectFunction(args.functionName)
    
    if not selected_function:
        print(f"Error: Function '{args.functionName}' not recognized.", file=sys.stderr)
        print("Available functions: logistic, sine, tent, henon", file=sys.stderr)
        sys.exit(1)

    # Handle initial state
    if is_multidimensional:
        # For Henon or other 2D maps, we need a tuple
        initial_state = (args.x0, 0.0)
    else:
        initial_state = args.x0

    if args.num_values <= 0:
        print("Error: num_values must be > 0", file=sys.stderr)
        sys.exit(1)

    generate_chaos_data(selected_function, initial_state, args.num_values, args.output_file)

if __name__ == "__main__":
    main()