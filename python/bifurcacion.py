import numpy as np
import matplotlib.pyplot as plt
import argparse

from modeloCaos import *

def bifurcation(function,r,r_min,r_max,num_r,iterations,transition,initial_condition,name,save,dpi,num_params=1):
    for n in range(0,num_params):

        r[n] = np.linspace(r_min[n],r_max[n], num_r[n])  # r values
        for i in range(0,num_params): # The rest of parameters are constant
            if i != n:
                r[i] = np.ones_like(r[n]) * r[i]

        x = initial_condition * np.ones_like(r[n])  # initial state

        for i in range(iterations):
            x = function(x,r)
            if i >= (transition):
                plt.plot(r[n], x[n], ',k', alpha=0.25)  # ',' = tiny point
        
        #plt.title("Bifurcation diagram - "+name)
        plt.xlabel("r")
        plt.ylabel("x")
        plt.grid(True)
        
        if save!="":
            plt.savefig(save, dpi=dpi, bbox_inches='tight')
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bifurcation diagram")
    parser.add_argument("functionName", help="Name of hte function", type=str)
    parser.add_argument("r_min", nargs="?",default=0, help="Lower end of r interval", type=float)
    parser.add_argument("r_max", nargs="?",default=4, help="Upper end of r interval", type=float)
    parser.add_argument("r_num", nargs="?",default=1000, help="Number of values of r", type=int)
    parser.add_argument("iterations", nargs="?",default=200, help="Total number of iterations", type=int)
    parser.add_argument("transition", nargs="?",default=100, help="Number of iterations skiped", type=int)
    parser.add_argument("initial_condition", nargs="?",default=1e-5, help="Initial condition", type=float)
    parser.add_argument("--save",default="", help="Path where to save the generated diagram.", type=str)
    parser.add_argument("--dpi",default=300, help="Densitiy of saved diagram.", type=int)
    args = parser.parse_args()
    r_min = args.r_min
    r_max = args.r_max
    r_num=args.r_num
    functionName=args.functionName
    iterations = args.iterations
    transition = args.transition
    initial_condition = args.initial_condition
    save=args.save
    dpi=args.dpi

    r=0

    function,num_params = selectFunction(functionName)
    if num_params == 1:
        r_max = [r_max]
        r_min = [r_min]
        r_num = [r_num]
        r = [0]
    
    bifurcation(function,r,r_min, r_max,r_num, iterations, transition, initial_condition,functionName,save,dpi,num_params)