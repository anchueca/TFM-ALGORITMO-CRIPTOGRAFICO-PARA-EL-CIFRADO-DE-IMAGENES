#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib

try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse

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


def plot(function, r, iterations, x0, y0, save, dpi):
    xs = [x0]
    ys = [y0]

    x = x0
    y = y0

    for _ in range(1, iterations):
        x = function(x, r)
        y = function(y, r)
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(10, 6))
    plt.scatter(range(iterations), xs, color='blue', label=f"x₀ = {x0}", s=15, alpha=0.7)
    plt.scatter(range(iterations), ys, color='red', label=f"y₀ = {y0}", s=15, alpha=0.7)

    plt.xlabel("Iteraciones")
    plt.ylabel("Valor x")
    plt.title(f"Sensibilidad a las Condiciones Iniciales (r = {r})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save:
        plt.savefig(save, dpi=dpi, bbox_inches='tight')
        print(f"Gráfico guardado en: {save}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparar dos trayectorias con condiciones iniciales ligeramente diferentes")
    parser.add_argument("functionName", nargs="?", default="logistic", type=str, help="Nombre de la función (ej: 'logistic', 'sine', 'tent', 'uno')")
    parser.add_argument("x0", nargs="?", default=0.4, type=float, help="Condición inicial x₀")
    parser.add_argument("y0", nargs="?", default=0.4000001, type=float, help="Condición inicial y₀ (ligeramente diferente)")
    parser.add_argument("r", nargs="?", default=3.9, type=float, help="Parámetro r del mapa")
    parser.add_argument("iterations", nargs="?", default=100, type=int, help="Número total de iteraciones")
    parser.add_argument("--save", default="", type=str, help="Ruta para guardar la imagen generada")
    parser.add_argument("--dpi", default=300, type=int, help="DPI del gráfico guardado")
    
    args = parser.parse_args()

    functionName = args.functionName
    iterations = args.iterations
    x0 = args.x0
    y0 = args.y0
    r = args.r
    save = args.save
    dpi = args.dpi

    function, num_params = selectFunction(functionName)
    if function is None:
        print(f"Error: Función '{functionName}' no encontrada.")
        sys.exit(1)

    plot(function, r, iterations, x0, y0, save, dpi)
