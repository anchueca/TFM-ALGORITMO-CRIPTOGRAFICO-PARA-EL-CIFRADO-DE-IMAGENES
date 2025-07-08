import numpy as np
import lyapynov
import argparse
import matplotlib.pyplot as plt
import jacobi

from modeloCaos import *

# Jacobiano del mapa logístico
def jac(x, t, r):
    #return np.array(r - 2 * r * x)  # Derivada de f(x)
    return np.array(r*np.pi*np.cos(np.pi*x*r))


# Función principal para calcular el exponente de Lyapunov
def calcular_exponentes(f,r_values, x0, iteraciones, puntos):
    lyap_values = []
    
    # Iterar sobre el intervalo de valores de r
    for r in r_values:
        # Crear el sistema discreto
        f_ajustada=lambda x, t: f(x,r)
        j_ajustado=lambda x, t: (jacobi.jacobi(lambda z: f_ajustada(z,t),x)[0])[0]
        discrete_system = lyapynov.DiscreteDS(x0, 1, f_ajustada, j_ajustado) #Más lento, pero automático
        #discrete_system = lyapynov.DiscreteDS(x0, 1, f_ajustada, lambda x, t: jac(x, t, r)) # Más rápidp, pero ay que ajustar el jacobiano
        #print(j_ajustado(x0,0)[0])
        #print(jac(x0,0,r))
        # Calcular los exponentes de Lyapunov
        result = lyapynov.LCE(discrete_system,len(x0), iteraciones, puntos, 0)
        
        # Extraer el exponente principal (suponiendo que el primero es el más relevante)
        lyap_values.append(result[0])
    
    return lyap_values

# Función para configurar los argumentos de línea de comandos
def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Calcular los exponentes de Lyapunov del mapa logístico.")
    parser.add_argument('function', type=str, help="Function to use")
    parser.add_argument('r_min', nargs="?", type=float, default=2.5, help="Valor mínimo del parámetro r")
    parser.add_argument('r_max', nargs="?", type=float, default=4.0, help="Valor máximo del parámetro r")
    parser.add_argument('r_steps', nargs="?", type=int, default=1000, help="Número de pasos en el intervalo de r")
    parser.add_argument('x0', nargs="?", type=float, default=0.5, help="Condición inicial del sistema")
    parser.add_argument('iteraciones', nargs="?", type=int, default=100, help="Número de iteraciones antes de calcular el exponente")
    parser.add_argument('puntos', nargs="?", type=int, default=1000, help="Número de puntos para calcular el exponente")
    parser.add_argument("--save",default="", help="Path where to save the generated diagram.", type=str)
    parser.add_argument("--dpi",default=300, help="Densitiy of saved diagram.", type=int)
    return parser.parse_args()

# Función principal para ejecutar el script
def main():
    # Obtener los argumentos de línea de comandos
    args = obtener_argumentos()

    f,num_args = selectFunction(args.function)

    # Crear un intervalo de valores para r
    r_values = np.linspace(args.r_min, args.r_max, args.r_steps)
    
    # Condición inicial
    x0 = np.array([args.x0])

    # Calcular los exponentes de Lyapunov para cada valor de r
    lyap_values = calcular_exponentes(f,r_values, x0, args.iteraciones, args.puntos)

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, lyap_values, label="Exponente de Lyapunov")
    plt.axhline(0, color='red', linestyle='--', label="Umbral de estabilidad")
    plt.xlabel("Valor de r")
    plt.ylabel("Exponente de Lyapunov")
    #plt.title("Exponentes de Lyapunov en función del parámetro r")
    plt.legend()
    plt.grid(True)

    save = args.save
    if save!="":
        plt.savefig(save, dpi=args.dpi, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    main()
