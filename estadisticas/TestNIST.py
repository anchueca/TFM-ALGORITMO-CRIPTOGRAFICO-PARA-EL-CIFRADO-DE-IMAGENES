import struct
import subprocess
import sys
import argparse
import os

from modeloCaos import *

def generate_chaos_data(function, x0, num_values, output_file=None):
    """
    Genera datos caóticos binarizados y los escribe en un archivo o los devuelve como bytes.

    Args:
        function (callable): Función caótica.
        x0 (float o lista): Condición inicial.
        num_values (int): Número de valores a generar.
        output_file (str, opcional): Archivo de salida. Si es None, devuelve los bytes generados.

    Returns:
        bytes o None: Bytes generados si no se especifica archivo, sino None.
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

        if output_file:
            f = open(output_file, 'wb')
            print(f"Escribiendo {num_values} valores en '{output_file}'...", file=sys.stderr)

        for n in range(num_values):
            x0 = function(x0)
            values = x0 if isinstance(x0, (list, tuple)) else [x0]

            for val in values:
                # Convertimos el float a 64 bits IEEE 754
                float_bits = struct.unpack('>Q', struct.pack('>d', val))[0]

                # Extraemos la mantisa de 52 bits
                mantissa = float_bits & ((1 << 52) - 1)

                # Tomamos los 32 bits más significativos de la mantisa
                top32 = mantissa >> (52 - 32)

                # Dividimos en 4 bloques de 8 bits y aplicamos XOR entre ellos
                b0 = (top32 >> 24) & 0xFF
                b1 = (top32 >> 16) & 0xFF
                b2 = (top32 >> 8)  & 0xFF
                b3 = top32 & 0xFF

                result_byte = b0 ^ b1 ^ b2 ^ b3  # XOR final

                write_byte(result_byte)

            if output_file and num_values > 10 and n % (num_values // 10) == 0:
                print(f"Progreso: {(n / num_values) * 100:.2f}%", file=sys.stderr)

        # Escribir bits restantes si no múltiplo de 8
        if bit_buffer:
            while len(bit_buffer) < 8:
                bit_buffer.append(0)
            byte = sum((bit << (7 - i)) for i, bit in enumerate(bit_buffer))
            write_byte(byte)

        if output_file:
            f.close()
            print("Progreso: 100.00%", file=sys.stderr)
            print(f"Escritura completada en '{output_file}'.", file=sys.stderr)
            return None
        else:
            print("Generación en memoria completada.", file=sys.stderr)
            return bytes(output_bytes)

    except Exception as e:
        print(f"Ocurrió un error inesperado durante la generación: {e}", file=sys.stderr)
        sys.exit(1)



def porTuberia(selected_function, initial_state,num_values,nist_command_base):
# Modo tubería: Generamos los bytes en memoria y los pasamos a subprocess.run.
    print("\nModo tubería: Generando datos y pasándolos directamente a NIST STS...", file=sys.stderr)
    generated_data_bytes = generate_chaos_data(selected_function, initial_state,num_values, output_file=None)

    if generated_data_bytes is None:
            print("Error: No se generaron datos para la tubería.", file=sys.stderr)
            sys.exit(1)

    nist_command = nist_command_base + ["-"] # '-' indica que sts lea de stdin
    try:
        print(f"Ejecutando NIST STS con entrada desde memoria ({len(generated_data_bytes)} bytes)...", file=sys.stderr)
        process = subprocess.run(
            nist_command,
            input=generated_data_bytes, # Pasa los bytes generados como entrada a sts
            capture_output=True,
            text=False # Decodifica stdout/stderr como texto
        )

        if process.returncode == 0:
            print("\n--- NIST STS - Salida Estándar ---", file=sys.stderr)
            print(process.stdout, file=sys.stderr)
            print("La suite NIST STS se ejecutó correctamente.", file=sys.stderr)
        else:
            print(f"\n--- NIST STS - Error (Código de salida: {process.returncode}) ---", file=sys.stderr)
            print(process.stderr, file=sys.stderr)
            print("La suite NIST STS falló.", file=sys.stderr)

    except Exception as e:
        print(f"Ocurrió un error al ejecutar la suite NIST STS: {e}", file=sys.stderr)
        sys.exit(1)



def porArchivo(selected_function, initial_state,num_values,nist_command_base, output_file):

    generate_chaos_data(selected_function, initial_state,num_values,output_file)

    if run_nist:
        # Si se generó un archivo y se pidió ejecutar NIST STS, lo hacemos.
        if not os.path.exists(output_file):
            print(f"Error: El archivo de salida '{output_file}' no se creó correctamente para NIST STS.", file=sys.stderr)
            sys.exit(1)

        nist_command = nist_command_base + [output_file]
        print(f"\nEjecutando la suite NIST STS en el archivo '{output_file}'...", file=sys.stderr)
        try:
            process = subprocess.run(
                nist_command,
                capture_output=True,
                text=True
            )

            if process.returncode == 0:
                print("\n--- NIST STS - Salida Estándar ---", file=sys.stderr)
                print(process.stdout, file=sys.stderr)
                print("La suite NIST STS se ejecutó correctamente.", file=sys.stderr)
            else:
                print(f"\n--- NIST STS - Error (Código de salida: {process.returncode}) ---", file=sys.stderr)
                print(process.stderr, file=sys.stderr)
                print("La suite NIST STS falló.", file=sys.stderr)

        except Exception as e:
            print(f"Ocurrió un error al ejecutar la suite NIST STS: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generar valores de una función caótica y, opcionalmente, analizarlos con la suite NIST STS.")
    parser.add_argument("functionName", help="Nombre de la función caótica (ej. logistic, sine, tent, superModelo)", type=str)
    parser.add_argument("x0", help="Condición inicial (número flotante)", type=float)
    parser.add_argument("num_values", help="Número de valores a generar", type=int)
    parser.add_argument("--output_file", help="Nombre del archivo de salida. Si no se especifica, los datos se pasan a NIST STS.", type=str)
    parser.add_argument("--run_nist", action="store_true", help="Ejecuta la suite NIST STS con los datos generados. Requiere --nist_path.")
    parser.add_argument("--nist_path", default="./nist/sts", help="Ruta al ejecutable de la suite NIST STS.")
    parser.add_argument("--nist_bits_per_sequence", type=int, default=1000000,
                        help="Número de bits por secuencia para el análisis NIST STS (-n). Default: 1,000,000.")
    args = parser.parse_args()

    selected_function = selectFunction(args.functionName)[0]
    if not selected_function:
        print(f"Error: Función '{args.functionName}' no reconocida.", file=sys.stderr)
        sys.exit(1)

    initial_state = args.x0

    num_values = args.num_values
    if num_values <= 0:
        print("Error: El número de valores a generar debe ser mayor que cero.", file=sys.stderr)
        sys.exit(1)

# --- 2. Preparación para NIST STS (si se solicita) ---
    
    nist_command_base = [
        args.nist_path,
        "-v", "1",
        "-i", 128,
        "-I", "1",
        "-w", ".", # Escribe los resultados en el directorio actual
        "-F", "r"  # Formato de entrada "raw"
    ]

    return selected_function,initial_state,num_values,nist_command_base, args.run_nist,args.output_file

if __name__ == "__main__":
    selected_function,initial_state,num_values,nist_command_base,run_nist,output_file = main()

    # --- 3. Generar datos y manejar la salida/tubería ---
    if not output_file:
        porTuberia(selected_function, initial_state,num_values,nist_command_base)
    else:
        porArchivo(selected_function, initial_state,num_values,nist_command_base, output_file)
