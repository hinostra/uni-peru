#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt
from modules.numerical_integration import *

import warnings
warnings.filterwarnings("ignore")

#### EJERCICIO 1 ####
def f1(x):
    return 1 - x - 4*x**3 + 2*x**5

print('Ejercicio 1')
print('Integral exacta: I = 1104')

a, b = -2, 4
print('Regla del trapecio: I = ', trapecio(f1, a, b))
for n in [2, 4]:
    print(f'Regla del trapecio compuesto (n={n}): I = ', trapecio(f1, a, b, n))
print('Regla de Simpson 1/3: I = ', simpson13(f1, a, b))
for n in [4, 8]:
    print(f'Regla de Simpson 1/3 compuesto (n={n}): I = ', simpson13(f1, a, b, n))
print('Regla de Simpson 3/8: I = ', simpson38(f1, a, b))
for n in [6, 9]:
    print(f'Regla de Simpson 3/8 compuesto (n={n}): I = ', simpson38(f1, a, b, n))

#### EJERCICIO 2 ####
t = array([1, 2, 3.25, 4.5, 6, 7, 8, 8.5, 9, 10])
v = array([5, 6, 5.5, 7, 8.5, 8, 6, 7, 7, 5])

print('Ejercicio 2')
print('La distancia recorrida por el auto es ', trapecio_noequid(t, v), ' m')

def f(x):
    return exp(x)*sin(x)/(1+x**2)
    # return 0.2 + 25*x -200*x**2 + 675*x**3 - 900*x**4 + 400*x**5

a, b = 0., 2.
I1 = romberg(f, a, b, 2., 1)
I2 = romberg(f, a, b, 2., 2)
e, n = abs(I1-I2)/abs(I2), 3
while e > 0.5:
    I1 = I2
    I2 = romberg(f, a, b, 2., n)
    n += 1
    e = abs(I1-I2)/abs(I2)*100
print('Ejercicio 3')
print(f'El valor de la integral es {I2} con {n} niveles')