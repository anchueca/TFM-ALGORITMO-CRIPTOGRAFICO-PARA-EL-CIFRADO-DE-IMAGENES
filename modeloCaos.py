import numpy as np

def uno(x,r=6.1):
    return np.abs( np.cos( r*np.cos( np.pi*(r+3*x*x) )*(r+3*x*x)*np.pi ) )

def logistic(x,r=3.8):
    return r*x*(1-x)

def sine(x,r=0.9):
    return np.sin(x*np.pi*r)

def tent(x,r):
    return np.where(x < 0.5, r*x, r*(1-x))

def selectFunction(functionName):
    if functionName == "logistic":
        return logistic,1
    elif functionName == "sine":
        return sine,1
    elif functionName == "tent":
        return tent,1
    elif functionName == "uno":
        return uno,1
    else:
        print("Function not recognized.")
        exit(1)