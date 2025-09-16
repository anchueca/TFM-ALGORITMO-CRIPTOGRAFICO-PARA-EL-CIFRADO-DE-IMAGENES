import numpy as np
import lyapynov
import argparse
import matplotlib.pyplot as plt
import jacobi

from modeloCaos import *

# Jacobian of the logistic map
def jac(x, t, r):
    #return np.array(r - 2 * r * x)  # Derivative of f(x)
    return np.array(r*np.pi*np.cos(np.pi*x*r))


# Main function to compute the Lyapunov exponent
def compute_exponents(f,r_values, x0, iterations, points):
    lyap_values = []
    
    # Iterate over the range of r values
    for r in r_values:
        # Create the discrete system
        adjusted_f = lambda x, t: f(x,r)
        adjusted_j = lambda x, t: (jacobi.jacobi(lambda z: adjusted_f(z,t),x)[0])[0]
        discrete_system = lyapynov.DiscreteDS(x0, 1, adjusted_f, adjusted_j) #Slower, but automatic
        #discrete_system = lyapynov.DiscreteDS(x0, 1, adjusted_f, lambda x, t: jac(x, t, r)) # Faster, but Jacobian must be adjusted
        #print(adjusted_j(x0,0)[0])
        #print(jac(x0,0,r))
        # Compute the Lyapunov exponents
        result = lyapynov.LCE(discrete_system,len(x0), iterations, points, 0)
        
        # Extract the principal exponent (assuming the first is the most relevant)
        lyap_values.append(result[0])
    
    return lyap_values

# Function to set command-line arguments
def get_arguments():
    parser = argparse.ArgumentParser(description="Compute the Lyapunov exponents of the logistic map.")
    parser.add_argument('function', type=str, help="Function to use")
    parser.add_argument('r_min', nargs="?", type=float, default=2.5, help="Minimum value of the parameter r")
    parser.add_argument('r_max', nargs="?", type=float, default=4.0, help="Maximum value of the parameter r")
    parser.add_argument('r_steps', nargs="?", type=int, default=1000, help="Number of steps in the r interval")
    parser.add_argument('x0', nargs="?", type=float, default=0.5, help="Initial condition of the system")
    parser.add_argument('iterations', nargs="?", type=int, default=100, help="Number of iterations before computing the exponent")
    parser.add_argument('points', nargs="?", type=int, default=1000, help="Number of points to compute the exponent")
    parser.add_argument("--save",default="", help="Path where to save the generated diagram.", type=str)
    parser.add_argument("--dpi",default=300, help="Density of saved diagram.", type=int)
    return parser.parse_args()

# Main function to run the script
def main():
    # Get the command-line arguments
    args = get_arguments()

    f,num_args = selectFunction(args.function)

    # Create a range of r values
    r_values = np.linspace(args.r_min, args.r_max, args.r_steps)
    
    # Initial condition
    x0 = np.array([args.x0])

    # Compute the Lyapunov exponents for each r value
    lyap_values = compute_exponents(f,r_values, x0, args.iterations, args.points)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, lyap_values, label="Lyapunov Exponent")
    plt.axhline(0, color='red', linestyle='--', label="Stability Threshold")
    plt.xlabel("Value of r")
    plt.ylabel("Lyapunov Exponent")
    #plt.title("Lyapunov Exponents as a Function of Parameter r")
    plt.legend()
    plt.grid(True)

    save = args.save
    if save!="":
        plt.savefig(save, dpi=args.dpi, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    main()
