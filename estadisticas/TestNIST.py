import struct
import subprocess
import sys
import argparse
import os

from modeloCaos import *

def generate_chaos_data(function, x0, num_values, data_format='d', output_file=None):
    """
    Generate chaotic data and writes it to a file or returns it as bytes

    Args:
        function (callable): chaotic function.
        x0 (float or list): Intial condition.
        num_values (int): El número de valores a generar.
        data_format (str): El formato de empaquetado ('d' para double, 'f' para float).
        output_file (str, optional): Si se especifica, los datos se escriben a este archivo.
                                     Si es None, la función devuelve los bytes generados.

    Returns:
        bytes or None: Los bytes generados si output_file es None, de lo contrario None.
    """
    generated_bytes = b'' # Para acumular bytes si la salida es en memoria

    try:
        if output_file:
            # Si se especifica un archivo, abrimos y escribimos directamente
            with open(output_file, 'wb') as f:
                print(f"Escribiendo {num_values} valores en '{output_file}'...", file=sys.stderr)
                for n in range(num_values):
                    x0 = function(x0)
                    if isinstance(x0, (list, tuple)):
                        for val in x0:
                            f.write(struct.pack(data_format, val))
                    else:
                        f.write(struct.pack(data_format, x0))

                    if num_values > 10 and n % (num_values // 10) == 0:
                        print(f"Progreso: {(n / num_values) * 100:.2f}%", file=sys.stderr)
                print("Progreso: 100.00%", file=sys.stderr)
            print(f"Escritura completada en '{output_file}'.", file=sys.stderr)
            return None # No se devuelven bytes si se escribió a un archivo
        else:
            # Si no hay archivo, generamos bytes en memoria
            print(f"Generando {num_values} valores en memoria para tubería...", file=sys.stderr)
            for n in range(num_values):
                x0 = function(x0)
                if isinstance(x0, (list, tuple)):
                    for val in x0:
                        generated_bytes += struct.pack(data_format, val)
                else:
                    generated_bytes += struct.pack(data_format, x0)
            print("Generación en memoria completada.", file=sys.stderr)
            return generated_bytes

    except struct.error as e:
        print(f"Error de formato al empaquetar datos: {e}. Asegúrate de que el formato '{data_format}' sea compatible con el tipo de datos generado.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la generación: {e}", file=sys.stderr)
        sys.exit(1)



def porTuberia(selected_function, initial_state,num_values,format,nist_command_base):
# Modo tubería: Generamos los bytes en memoria y los pasamos a subprocess.run.
    print("\nModo tubería: Generando datos y pasándolos directamente a NIST STS...", file=sys.stderr)
    generated_data_bytes = generate_chaos_data(selected_function, initial_state,num_values,format, output_file=None)

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



def porArchivo(selected_function, initial_state,num_values,format,nist_command_base, output_file):

    generate_chaos_data(selected_function, initial_state,num_values,format,output_file)

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
    parser.add_argument("--format", default='d', help="Formato de empaquetado ('d' para double, 'f' para float). Default: 'd'", type=str)
    parser.add_argument("--run_nist", action="store_true", help="Ejecuta la suite NIST STS con los datos generados. Requiere --nist_path.")
    parser.add_argument("--nist_path", default="./nist/sts", help="Ruta al ejecutable de la suite NIST STS.")
    parser.add_argument("--nist_bits_per_sequence", type=int, default=1000000,
                        help="Número de bits por secuencia para el análisis NIST STS (-n). Default: 1,000,000.")
    args = parser.parse_args()

    # --- 1. Mapeo de función y estado inicial ---
    function_map = {
        "logistic": logistic,
        "sine": sine,
        "tent": tent,
        "superModelo": superModelo
    }

    selected_function = function_map.get(args.functionName)
    if not selected_function:
        print(f"Error: Función '{args.functionName}' no reconocida.", file=sys.stderr)
        sys.exit(1)

    initial_state = args.x0
    if args.functionName == "superModelo":
        initial_state = [args.x0, args.x0, args.x0] # superModelo espera un vector

    num_values = args.num_values
    if num_values <= 0:
        print("Error: El número de valores a generar debe ser mayor que cero.", file=sys.stderr)
        sys.exit(1)

# --- 2. Preparación para NIST STS (si se solicita) ---
    nist_bits_per_value = None
    data_format= args.format
    if data_format == 'd':
        nist_bits_per_value = "64"
    elif data_format == 'f':
        nist_bits_per_value = "32"
    else:
        print(f"Advertencia: Formato '{data_format}' no reconocido para NIST STS. Usando 32 bits por defecto.", file=sys.stderr)
        nist_bits_per_value = "32"

    total_generated_bits = num_values * int(nist_bits_per_value)
    num_sequences_for_nist = total_generated_bits // args.nist_bits_per_sequence
    nist_command_base = [
        args.nist_path,
        "-v", "1",
        "-i", nist_bits_per_value,
        "-I", "1",
        "-w", ".", # Escribe los resultados en el directorio actual
        "-F", "r"  # Formato de entrada "raw"
    ]

    return selected_function,initial_state,num_values,nist_command_base, args.run_nist,args.output_file,data_format

if __name__ == "__main__":
    selected_function,initial_state,num_values,nist_command_base,run_nist,output_file,data_format = main()

    # --- 3. Generar datos y manejar la salida/tubería ---
    if not output_file:
        porTuberia(selected_function, initial_state,num_values,data_format,nist_command_base)
    else:
        porArchivo(selected_function, initial_state,num_values,data_format,nist_command_base, output_file)
