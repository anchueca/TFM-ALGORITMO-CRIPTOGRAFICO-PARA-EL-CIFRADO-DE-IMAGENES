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


def bifurcation(function, r, r_min, r_max, num_r, iterations, transition, initial_condition, name, save, dpi, num_params=1):
    for n in range(0, num_params):
        r[n] = np.linspace(r_min[n], r_max[n], num_r[n])  # r values
        for i in range(0, num_params):  # The rest of parameters are constant
            if i != n:
                r[i] = np.ones_like(r[n]) * r[i]

        x = initial_condition * np.ones_like(r[n])  # initial state

        for i in range(iterations):
            x = function(x, r)
            if i >= transition:
                plt.plot(r[n], x[n], ',k', alpha=0.25)  # ',' = tiny point

        plt.title(f"Diagrama de Bifurcación - {name.capitalize()}")
        plt.xlabel("Parámetro r")
        plt.ylabel("Estado x")
        plt.grid(True, alpha=0.3)

        if save != "":
            plt.savefig(save, dpi=dpi, bbox_inches='tight')
            print(f"Diagrama guardado en: {save}")
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagrama de bifurcación para mapas 1D")
    parser.add_argument("functionName", nargs="?", default="logistic", help="Nombre de la función (ej: 'logistic', 'sine', 'tent', 'uno')", type=str)
    parser.add_argument("r_min", nargs="?", default=2.5, help="Extremo inferior del intervalo r", type=float)
    parser.add_argument("r_max", nargs="?", default=4.0, help="Extremo superior del intervalo r", type=float)
    parser.add_argument("r_num", nargs="?", default=1000, help="Número de valores de r", type=int)
    parser.add_argument("iterations", nargs="?", default=200, help="Número total de iteraciones", type=int)
    parser.add_argument("transition", nargs="?", default=100, help="Iteraciones transitorias descartadas", type=int)
    parser.add_argument("initial_condition", nargs="?", default=1e-5, help="Condición inicial", type=float)
    parser.add_argument("--save", default="", help="Ruta para guardar el diagrama generado.", type=str)
    parser.add_argument("--dpi", default=300, help="Densidad DPI del gráfico guardado.", type=int)
    
    args = parser.parse_args()
    
    r_min = args.r_min
    r_max = args.r_max
    r_num = args.r_num
    functionName = args.functionName
    iterations = args.iterations
    transition = args.transition
    initial_condition = args.initial_condition
    save = args.save
    dpi = args.dpi

    r = 0
    function, num_params = selectFunction(functionName)
    if function is None:
        print(f"Error: Función '{functionName}' no encontrada.")
        sys.exit(1)

    if num_params == 1:
        r_max = [r_max]
        r_min = [r_min]
        r_num = [r_num]
        r = [0]

    bifurcation(function, r, r_min, r_max, r_num, iterations, transition, initial_condition, functionName, save, dpi, num_params)
