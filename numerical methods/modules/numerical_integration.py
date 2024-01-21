#! /usr/bin/python3
from numpy import *

import warnings
warnings.filterwarnings("ignore")

''' MÉTODO DEL TRAPECIO '''
# f: función
# a, b: intervalo
# n: número de subintervalos (x0, x1, ..., xn)
def trapecio(f, a, b, n=1):
    h = (b-a)/n
    I = h/2*(f(a)+f(b))
    for k in range(1, n):
        I += h*f(a+k*h)     # 2*f(xi)/(2*n)
    return I

def trapecio_noequid(x, y):
    n = len(x)
    I = 0
    for i in range(n-1):
        I += (x[i+1] - x[i])/2*(y[i+1] + y[i])
    return I

''' MÉTODO DE SIMPSON 1/3 '''
# f: función
# a, b: intervalo
# n: número 2*k de subintervalos (x0, x1, ..., xn)
def simpson13(f, a, b, n=2):
    h = (b-a)/n
    I = h/3*(f(a)+4*f(a+h)+f(b))
    for k in range(2, n, 2):
        I += h/3*(2*f(a+k*h) + 4*f(a+(k+1)*h))
    return I

''' MÉTODO DE SIMPSON 3/8 '''
# f: función
# a, b: intervalo
# n: número 3*k de subintervalos (x0, x1, ..., xn)
def simpson38(f, a, b, n=3):
    h = (b-a)/n
    I = 3*h/8*(f(a)+3*f(a+h)+3*f(a+2*h)+f(b))
    for k in range(3, n, 3):
        I += 3*h/8*(2*f(a+k*h) + 3*f(a+(k+1)*h) + 3*f(a+(k+2)*h))
    return I

''' EXTRAPOLACIÓN DE RICHARDSON'''
# Iw, Ib: Integrales aproximadas
# k: nivel
def richardson(Iw, Ib, k=1):
    Cb, Cw = 4**k/(4**k-1), 1/(4**k-1)
    return Cb*Ib - Cw*Iw

''' ALGORITMO DE ROMBERG '''
# f: función
# a, b: intervalo
# k: nivel de la extrapolación
def romberg(f, a, b, h1, n=1):
    # Nivel 0
    I0 = list()
    for k in range(n+1):
        I0.append(trapecio(f, a, b, 2**k*int((b-a)/h1)))
    
    I2 = array(I0)
    for i in range(n):
        I1 = copy(I2)
        for j in range(n-i):
            I2[j] = richardson(I1[j], I1[j+1], i+1)

    return I2[0]

''' RAÍCES Y PESOS PARA CUADRATURA DE GAUSS-LEGENDRE'''
# n: número de puntos
def rw_gauss_legendre(n):
    if n==1: 
        roots = array([0], float)
        weights = array([2], float)
    elif n==2:
        roots = array([-1/sqrt(3), 1/sqrt(3)], float)
        weights = array([1, 1], float)
    elif n==3:
        roots = array([-sqrt(3/5), 0, sqrt(3/5)], float)
        weights = array([5/9, 8/9, 5/9], float)
    elif n==4:
        roots = array([-sqrt(3/7+2/7*sqrt(6/5)), -sqrt(3/7-2/7*sqrt(6/5)), sqrt(3/7-2/7*sqrt(6/5)), sqrt(3/7+2/7*sqrt(6/5))], float)
        weights = array([(18-sqrt(30))/36, (18+sqrt(30))/36, (18+sqrt(30))/36, (18-sqrt(30))/36], float)
    elif n==5:
        roots = array([-1/3*sqrt(5+2*sqrt(10/7)), -1/3*sqrt(5-2*sqrt(10/7)), 0, 1/3*sqrt(5-2*sqrt(10/7)), 1/3*sqrt(5+2*sqrt(10/7))], float)
        weights = array([(322-13*sqrt(70))/900, (322+13*sqrt(70))/900, 128/225, (322+13*sqrt(70))/900, (322-13*sqrt(70))/900], float)
    else:
        print('Ingrese otro valor de n.')
    return roots, weights

''' RAÍCES Y PESOS PARA CUADRATURA DE GAUSS-RANDAU-LEGENDRE '''
# n: número de puntos
def rw_gauss_radau_legendre(n):
    if n==3:
        roots = array([-1.0, -0.289898, 0.689898])
        weights = array([0.222222, 1.0249717, 0.7528061])
    elif n==4:
        roots = array([-1.0, -0.575319, 0.181066, 0.822824])
        weights = array([0.125, 0.657689, 0.776387, 0.440924])
    elif n==5:
        roots = array([-1.0, -0.720480, -0.167181, 0.446314, 0.885792])
        weights = array([0.08, 0.446208, 0.623653, 0.562712, 0.287427])
    else:
        print('Ingrese otro valor de n.')
    return roots, weights

''' RAÍCES Y PESOS PARA CUADRATURA DE GAUSS-LOBATTO-LEGENDRE '''
# n: número de puntos
def rw_gauss_lobatto_legendre(n):
    if n==2:
        roots = array([-1.0, 1.0])
        weights = array([1.0, 1.0])
    elif n==3:
        roots = array([-1.0, 0.0, 1.0])
        weights = array([1/3, 4/3, 1/3], float)
    elif n==4:
        roots = array([-1.0, -0.447213595499958, 0.447213595499958, 1.0])
        weights = array([1/6, 5/6, 5/6, 1/6], float)
    elif n==5:
        roots = array([-1.0, -0.654653670707977, 0.0, 0.654653670707977, 1.0])
        weights = array([0.1, 49/90, 32/45, 49/90, 0.1], float)
    elif n==6:
        roots = array([-1.0, -0.765055323929465, -0.285231516480645, 0.285231516480645, 0.765055323929465, 1.0])
        weights = array([1/15, 0.378474956297847, 0.554858377035486], float)
    elif n==7:
        roots = array([-1.0, -0.830223896278567, -0.468848793470714, 0.0, 0.468848793470714, 0.830223896278567, 1.0])
        weights = array([1/21, 0.276826047361566, 0.431745381209863, 0.487619047619048, 0.431745381209863, 0.276826047361566, 1/21], float)
    elif n==8:
        roots = array([-1.0, -0.871740148509607, -0.591700181433142, -0.209299217902479, 0.209299217902479, 0.591700181433142, 0.871740148509607, 1.0])
        weights = array([0.035714285714286, 0.210704227143506, 0.341122692483504, 0.412458794658704, 0.412458794658704, 0.341122692483504, 0.210704227143506, 0.035714285714286], float)
    else:
        print('Ingrese otro valor de n.')
    return roots, weights

''' CUADRATURA '''
# f: función
# a, b: intervalo de integración
# n: número de puntos
# tipo: tipo de cuadratura
def gauss_quad(f, a, b, n, tipo='Gauss-Legendre'):
    if tipo=='Gauss-Legendre':
        x, w = rw_gauss_legendre(n)
    elif tipo=='Gauss-Radau-Legendre':
        x, w = rw_gauss_radau_legendre(n)
    elif tipo=='Gauss-Lobatto-Legendre':
        x, w = rw_gauss_lobatto_legendre(n)
    else:
        print('Tipo incorrecto')
    # print(f'Para la cuadratura de {tipo} con {n} puntos:')
    # print('Las raíces son: ', x)
    # print('Los pesos son: ', w)
    I = 0
    for j in range(n):
        I += (b-a)/2*w[j]*f((b-a)/2*x[j] + (b+a)/2)
    return I