#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib

try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from chaos_generator.Chaos_Generator import selectFunction
except ImportError:
    from Chaos_Generator import selectFunction

try:
    import lyapynov
    import jacobi
    HAS_LYAPYNOV = True
except ImportError:
    HAS_LYAPYNOV = False


# Jacobian of the logistic map
def jac(x, t, r):
    return np.array(r * np.pi * np.cos(np.pi * x * r))


def compute_exponents(f, r_values, x0, iterations, points):
    lyap_values = []
    
    for r in r_values:
        if HAS_LYAPYNOV:
            adjusted_f = lambda x, t: f(x, r)
            adjusted_j = lambda x, t: (jacobi.jacobi(lambda z: adjusted_f(z, t), x)[0])[0]
            discrete_system = lyapynov.DiscreteDS(x0, 1, adjusted_f, adjusted_j)
            result = lyapynov.LCE(discrete_system, len(x0), iterations, points, 0)
            lyap_values.append(result[0])
        else:
            # Fallback derivative-based numerical Lyapunov Exponent for 1D maps
            x = x0[0]
            for _ in range(50): # Transient
                x = f(x, r)
            le_sum = 0.0
            dx = 1e-7
            for _ in range(points):
                df = (f(x + dx, r) - f(x - dx, r)) / (2 * dx)
                if abs(df) > 1e-12:
                    le_sum += np.log(abs(df))
                x = f(x, r)
            lyap_values.append(le_sum / points)

    return lyap_values


def get_arguments():
    parser = argparse.ArgumentParser(description="Calcular los exponentes de Lyapunov para mapas 1D.")
    parser.add_argument('function', nargs="?", type=str, default="logistic", help="Función caótica (ej: 'logistic', 'sine', 'tent', 'uno')")
    parser.add_argument('r_min', nargs="?", type=float, default=2.5, help="Valor mínimo de r")
    parser.add_argument('r_max', nargs="?", type=float, default=4.0, help="Valor máximo de r")
    parser.add_argument('r_steps', nargs="?", type=int, default=500, help="Número de pasos en el intervalo r")
    parser.add_argument('x0', nargs="?", type=float, default=0.5, help="Condición inicial del sistema")
    parser.add_argument('iterations', nargs="?", type=int, default=100, help="Número de iteraciones antes del cálculo")
    parser.add_argument('points', nargs="?", type=int, default=500, help="Número de puntos para promediar exponente")
    parser.add_argument("--save", default="", help="Ruta para guardar el diagrama generado.", type=str)
    parser.add_argument("--dpi", default=300, help="Densidad DPI del diagrama guardado.", type=int)
    return parser.parse_args()


def main():
    args = get_arguments()

    f, num_args = selectFunction(args.function)
    if f is None:
        print(f"Error: Función '{args.function}' no encontrada.")
        sys.exit(1)

    r_values = np.linspace(args.r_min, args.r_max, args.r_steps)
    x0 = np.array([args.x0])

    print(f"Calculando Exponente de Lyapunov para '{args.function}' (r ∈ [{args.r_min}, {args.r_max}])...")
    lyap_values = compute_exponents(f, r_values, x0, args.iterations, args.points)

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, lyap_values, label=f"Exponente de Lyapunov ({args.function.capitalize()})")
    plt.axhline(0, color='red', linestyle='--', label="Umbral de Estabilidad (LE=0)")
    plt.xlabel("Valor de r")
    plt.ylabel("Exponente de Lyapunov")
    plt.title(f"Exponente de Lyapunov vs Parámetro r — Mapa {args.function.capitalize()}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if args.save:
        plt.savefig(args.save, dpi=args.dpi, bbox_inches='tight')
        print(f"Gráfico guardado en: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
