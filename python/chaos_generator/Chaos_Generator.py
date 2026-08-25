#!/usr/bin/env python3
import os
import sys
import numpy as np
import math
import argparse
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


def uno(x, r=6.1):
    return np.abs(np.cos(r * np.cos(np.pi * (r + 3 * x * x)) * (r + 3 * x * x) * np.pi))


def logistic_map(x, r=3.9999):
    """
    Logistic Map: x_n+1 = r * x_n * (1 - x_n)
    Standard chaotic parameter r approx 4.0
    """
    return r * x * (1 - x)


def tent_map(x, mu=1.9999):
    """
    Tent Map: x_n+1 = mu * min(x_n, 1 - x_n)
    """
    return mu * min(x, 1 - x)


def sine_map(x, a=0.9999):
    """
    Sine Map: x_n+1 = a * sin(pi * x_n)
    """
    return a * math.sin(math.pi * x)


def henon_map(point, a=1.4, b=0.3):
    """
    Henon Map (2D): 
    x_n+1 = 1 - a * x_n^2 + y_n
    y_n+1 = b * x_n
    Input 'point' is a tuple/list (x, y). Returns (new_x, new_y).
    """
    x, y = point
    new_x = 1 - a * x**2 + y
    new_y = b * x
    return new_x, new_y


def selectFunction(name):
    """
    Selects the chaotic function based on string name.
    Returns a tuple: (function, is_multidimensional)
    """
    name = name.lower()
    if name in ["logistic", "logístico", "logistico"]:
        return logistic_map, False
    elif name == "tent":
        return tent_map, False
    elif name == "sine":
        return sine_map, False
    elif name == "henon":
        return henon_map, True
    elif name == "uno":
        return uno, False
    else:
        return None, False


def main():
    parser = argparse.ArgumentParser(description="Simulador de Mapas Caóticos")
    parser.add_argument("--map", type=str, default="all", choices=["all", "logistic", "tent", "sine", "henon", "uno"],
                        help="Mapa caótico a simular (default: all)")
    parser.add_argument("--iterations", type=int, default=100, help="Número de iteraciones (default: 100)")
    parser.add_argument("--x0", type=float, default=0.4, help="Condición inicial x0 (default: 0.4)")
    parser.add_argument("--save", type=str, default="", help="Ruta para guardar la imagen PNG (opcional)")
    parser.add_argument("--dpi", type=int, default=300, help="Resolución de la imagen guardada")

    args = parser.parse_args()

    plt.figure(figsize=(10, 6))

    if args.map == "all":
        # Simulate and plot 1D maps together
        maps = [("Logistic", logistic_map), ("Tent", tent_map), ("Sine", sine_map), ("Uno", uno)]
        for name, func in maps:
            x_vals = [args.x0]
            x = args.x0
            for _ in range(args.iterations - 1):
                x = func(x)
                x_vals.append(x)
            plt.plot(range(args.iterations), x_vals, label=f"{name} Map", alpha=0.8, linewidth=1.5)
        plt.title(f"Comparativa de Mapas Caóticos (x0={args.x0})", fontsize=14)
        plt.xlabel("Iteración")
        plt.ylabel("Estado x")
        plt.legend()
        plt.grid(True, alpha=0.3)
    elif args.map == "henon":
        x_vals, y_vals = [args.x0], [args.x0]
        p = (args.x0, args.x0)
        for _ in range(args.iterations - 1):
            p = henon_map(p)
            x_vals.append(p[0])
            y_vals.append(p[1])
        plt.scatter(x_vals, y_vals, s=5, c='purple', alpha=0.6)
        plt.title(f"Atractor de Hénon ({args.iterations} iteraciones)", fontsize=14)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
    else:
        func, _ = selectFunction(args.map)
        x_vals = [args.x0]
        x = args.x0
        for _ in range(args.iterations - 1):
            x = func(x)
            x_vals.append(x)
        plt.plot(range(args.iterations), x_vals, 'b-o', markersize=3, label=f"{args.map.capitalize()} Map")
        plt.title(f"Evolución del Mapa {args.map.capitalize()} (x0={args.x0})", fontsize=14)
        plt.xlabel("Iteración")
        plt.ylabel("Estado x")
        plt.legend()
        plt.grid(True, alpha=0.3)

    if args.save:
        plt.savefig(args.save, dpi=args.dpi, bbox_inches='tight')
        print(f"Gráfico guardado en: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
