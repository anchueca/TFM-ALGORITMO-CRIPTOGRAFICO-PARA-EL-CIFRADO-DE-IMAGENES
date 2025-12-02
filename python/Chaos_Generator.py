import numpy as np
import math

def uno(x,r=6.1):
    return np.abs( np.cos( r*np.cos( np.pi*(r+3*x*x) )*(r+3*x*x)*np.pi ) )

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
    if name == "logistic":
        return logistic_map, False
    elif name == "tent":
        return tent_map, False
    elif name == "sine":
        return sine_map, False
    elif name == "henon":
        return henon_map, True
    else:
        return None, False