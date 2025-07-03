import numpy as np

def superModelo(x,r):
    return [
        logistic(r[0],x[2])/3 + sine(r[1],x[0])/3 + tent(r[2],x[1])/3,
        logistic(r[0],x[0])/3 + sine(r[1],x[1])/3 + tent(r[2],x[2])/3,
        logistic(r[0],x[1])/3 + sine(r[1],x[2])/3 + tent(r[2],x[0])/3,
    ]

def uno(x,r=0.6):
    return  np.abs( np.cos( np.pi*r*np.cos( np.pi*(r+3*x*x) )*(r+3*x*x) ) )

def logistic(x,r=3.8):
    return r*x*(1-x)

def sine(x,r=0.9):
    return np.sin(x*np.pi*r)

def tent(x,r):
    return np.where(x < 0.5, r*x, r*(1-x))

def gauss(x,r):
    if r is not list:
        r = [6, r]
    return np.exp(-r[0]*x*x) + r[1]

def selectFunction(functionName):
    if functionName == "logistic":
        return logistic,1
    elif functionName == "sine":
        return sine,1
    elif functionName == "tent":
        return tent,1
    elif functionName == "superModelo":
        return superModelo,3
    elif functionName == "gauss":
        return gauss,1
    elif functionName == "uno":
        return uno,1
    else:
        print("Function not recognized.")
        exit(1)