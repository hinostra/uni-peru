#! /usr/bin/python3
from numpy import *

import warnings
warnings.filterwarnings("ignore")

''' DIFERENCIACIÓN FORWARD '''
# f: función o datos f(x)
# i: punto donde se evalúa la derivada
# h: ancho de paso
# order: 1, 2, 3, 4
def der_forward(f, i, h, order=1):
    if order==1:
        df = (f[i+1] - f[i])/h
    elif order==2:
        df = (f[i+2] - 2*f[i+1] + f[i])/h**2
    elif order==3:
        df = (f[i+3] - 3*f[i+2] + 3*f[i+1] - f[i])/h**3
    elif order==4:
        df = (f[i+4] - 4*f[i+3] + 6*f[i+2] - 4*f[i+1] + f[i])/h**4
    else:
        print('Orden incorrecto')
    return df

''' DIFERENCIACIÓN BACKWARD '''
# f: función o datos f(x)
# i: punto donde se evalúa la derivada
# h: ancho de paso
# order: 1, 2, 3, 4
def der_backward(f, i, h, order=1):
    if order==1:
        df = (f[i] - f[i-1])/h
    elif order==2:
        df = (f[i] - 2*f[i-1] + f[i-2])/h**2
    elif order==3:
        df = (f[i] - 3*f[i-1] + 3*f[i-2] - f[i-3])/h**3
    elif order==4:
        df = (f[i] - 4*f[i-1] + 6*f[i-2] - 4*f[i-3] + f[i-4])/h**4
    else:
        print('Orden incorrecto')
    return df

''' DIFERENCIACIÓN CENTRADA '''
# f: función o datos f(x)
# i: punto donde se evalúa la derivada
# h: ancho de paso
# order: 1, 2, 3, 4
def der_centered(f, i, h, order=1):
    if order==1:
        df = (f[i+1] - f[i-1])/(2*h)
    elif order==2:
        df = (f[i+1] - 2*f[i] + f[i-1])/h**2
    elif order==3:
        df = (f[i+2] - 2*f[i+1] + 2*f[i-1] - f[i-2])/2*h**3
    elif order==4:
        df = (f[i+2] - 4*f[i+1] + 6*f[i] - 4*f[i-1] + f[i-2])/h**4
    else:
        print('Orden incorrecto')
    return df


''' EXTRAPOLACIÓN DE RICHARDSON'''
# Dw, Db: Derivadas aproximadas
# k: nivel
def richardson(Dw, Db, k=1):
    Cb, Cw = 4**k/(4**k-1), 1/(4**k-1)
    return Cb*Db - Cw*Dw

''' ALGORITMO DE ROMBERG PARA DERIVADAS '''
# f: función
# i: punto donde se evalúa la derivada
# h1: ancho de paso
# k: nivel de la extrapolación
def der_extrapolation(f, i, h1, n=1):
    D0 = list()
    for k in range(n+1):
        D0.append(der_centered(f, i, h1/2**k))
    
    D2 = array(D0)
    for i in range(n):
        D1 = copy(D2)
        for j in range(n-i):
            D2[j] = richardson(D1[j], D1[j+1], i+1)

    return D2[0]

''' DERIVADAS PARCIALES '''
# u: función de x, y
# i, j: puntos donde se evalúa la derivada
# h, k: ancho de paso para x, y
# n, m: orden de la derivada en x, y
def der_partial(u, i, j, h, k, n=1, m=0):
    if n==1 and m==0:
        du = der_centered(u[:,j], i, h, 1)
    if n==0 and m==1:
        du = der_centered(u[i,:], j, k, 1)
    if n==2 and m==0:
        du = der_centered(u[:,j], i, h, 2)
    if n==0 and m==2:
        du = der_centered(u[i,:], j, k, 2)
    if n==1 and m==1:
        du = (-u[i-1,j+1] + u[i+1,j+1] + u[i-1,j-1] - u[i+1,j-1])/(4*h**2)
    else:
        print('No disponible.')

''' LAPLACIANO '''
# u: función de x, y
# i, j: puntos donde se evalúa la derivada
# h, k: ancho de paso para x, y
# laplace: nabla^2 u (True), nabla^4 (False)
def laplacian(u, i, j, h, laplace=True):
    if laplace:
        du = (u[i,j+1] + u[i-1,j] -4*u[i,j] + u[i+1,j] + u[i,j-1])/h**2
    else:
        du = (u[i,j+2] + 2*u[i-1,j+1] - 8*u[i,j+1] + 2*u[i+1,j+1] + u[i-2,j] - 8*[i-1,j] + 20*u[i,j] - 8*u[i+1,j] + u[i+2,j] + u[i-1,j-1] - 8*u[i,j-1] + 2*u[i+1,j-1] + u[i,j-2])/h**4
    return du