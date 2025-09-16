import numpy as np
import matplotlib.pyplot as plt
import argparse

from modeloCaos import *

def plot(function,r,iterations,x0,y0,save,dpi):
    xs = [x0]
    ys = [y0]
    
    x = x0
    y = y0
    
    for _ in range(1, iterations):
        x = function(x, r)
        y = function(y, r)
        xs.append(x)
        ys.append(y)
        print(x)

    plt.scatter(range(iterations), xs, color='blue', label=f"x₀ = {x0}")
    plt.scatter(range(iterations), ys, color='red', label=f"y₀ = {y0}")
    
    plt.xlabel("Iteraciones")
    plt.ylabel("Valor")
    plt.title(f"Sensibilidad a condiciones iniciales (r = {r})")
    plt.legend()
    plt.grid(True)
    
    if save:
        plt.savefig(save, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two trajectories with different initial conditions")
    parser.add_argument("functionName", type=str, help="Name of the function (e.g. 'logistic')")
    parser.add_argument("x0", nargs="?", default=0.4, type=float, help="Initial condition x₀")
    parser.add_argument("y0", nargs="?", default=0.4000001, type=float, help="Initial condition y₀ (slightly different)")
    parser.add_argument("r", nargs="?", default=3.9, type=float, help="Parameter r for the map")
    parser.add_argument("iterations", nargs="?", default=200, type=int, help="Total number of iterations")
    parser.add_argument("--save", default="", type=str, help="Path to save the generated plot")
    parser.add_argument("--dpi", default=300, type=int, help="DPI (resolution) for the saved plot")
    args = parser.parse_args()

    functionName=args.functionName
    iterations = args.iterations
    x0 = args.x0
    y0 = args.y0
    r = args.r
    save=args.save
    dpi=args.dpi

    function,num_params = selectFunction(functionName)
    
    plot(function,r,iterations,x0,y0,save,dpi)