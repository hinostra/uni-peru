#! /usr/bin/python3
from random import gauss
from numpy import *
import matplotlib.pyplot as plt
from modules.numerical_integration import *
from modules.numerical_differentiation import *

import warnings
warnings.filterwarnings("ignore")

# def f(x):
#     return cos(x)

# a, b = 0., pi/2
# print('GL: I = ', gauss_quad(f, a, b, 3, 'Gauss-Legendre'))
# print('GRL: I = ', gauss_quad(f, a, b, 3, 'Gauss-Radau-Legendre'))
# print('GLL: I = ', gauss_quad(f, a, b, 3, 'Gauss-Lobatto-Legendre'))

#### EJERCICIO 1 ####
def f1(x):
    return 4/(1+x**2)

a, b = 0., 1.

print('1. Cálculo de pi')
print(pi)
print('Cuadratura de Gauss-Legendre')
print('\t[n=3] pi = ', gauss_quad(f1, a, b, 3, 'Gauss-Legendre'))
print('\t[n=4] pi = ', gauss_quad(f1, a, b, 4, 'Gauss-Legendre'))
print('\t[n=5] pi = ', gauss_quad(f1, a, b, 5, 'Gauss-Legendre'))

print('Cuadratura de Gauss-Randau-Legendre')
print('\t[n=3] pi = ', gauss_quad(f1, a, b, 3, 'Gauss-Radau-Legendre'))
print('\t[n=4] pi = ', gauss_quad(f1, a, b, 4, 'Gauss-Radau-Legendre'))
print('\t[n=5] pi = ', gauss_quad(f1, a, b, 5, 'Gauss-Radau-Legendre'))

print('Cuadratura de Gauss-Lobato-Legendre')
print('\t[n=3] pi = ', gauss_quad(f1, a, b, 3, 'Gauss-Lobatto-Legendre'))
print('\t[n=4] pi = ', gauss_quad(f1, a, b, 4, 'Gauss-Lobatto-Legendre'))
print('\t[n=5] pi = ', gauss_quad(f1, a, b, 5, 'Gauss-Lobatto-Legendre'))

#### EJERCICIO 2 ####
t = array([200, 202, 204, 206, 208, 210], float)
h = 2.
theta = array([0.75, 0.72, 0.70, 0.68, 0.67, 0.66])
r = array([5120, 5370, 5560, 5800, 6030, 6240], float)

# Derivadas de r y theta

rd, rdd = zeros(6), zeros(6)
thetad, thetadd = zeros(6), zeros(6)

rd[0] = der_forward(r, 0, h)
rdd[0] = der_forward(r, 0, h, 2)
thetad[0] = der_forward(theta, 0, h)
thetadd[0] = der_forward(theta, 0, h, 2)
rd[-1] = der_backward(r, -1, h)
rdd[-1] = der_backward(r, -1, h, 2)
thetad[-1] = der_backward(theta, -1, h)
thetadd[-1] = der_backward(theta, -1, h, 2)

for i in range(1, 5):
    rd[i] = der_centered(r, i, h)
    rdd[i] = der_centered(r, i, h, 2)
    thetad[i] = der_backward(theta, i, h)
    thetadd[i] = der_backward(theta, i, h, 2)

for i in range(6):
    vr = rd[i]
    vtheta = r[i]*thetad[i]
    ar = rdd[i] - r[i]*thetad[i]**2
    atheta = r[i]*thetadd[i] + 2*rd[i]*thetad[i]
    print(f'Para t = {t[i]} s')
    print(f'\tv = {vr: .2f} er + {vtheta: .2f} e0 m/s')
    print(f'\ta = {ar: .2f} er + {atheta: .2f} e0 m/s^2')