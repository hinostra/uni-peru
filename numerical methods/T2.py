#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt

import warnings

from sympy import false
warnings.filterwarnings("ignore")

from modules.nonlinear_eq import *

# Función f(x) = x - 5 + 5e^x = 0
def f(x):
    return x - 5 + 5*exp(-x) 

# Función g(x) = x para Punto Fijo
def g(x):
    return 5*(1 - exp(-x))

# Derivada de f(x) para Newton-Raphson
def df(x):
    return 1 - 5*exp(-x)

# Plot para restringir rango de búsqueda
xtest = linspace(0, 6, 100)
plt.plot(xtest, f(xtest))
plt.title(r'$f(x) = x+5e^{-x}-5$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()

# tol: Precisión de x 1e-3% (para que la precisión de b sea 1e-6)
# Nmax: Número máximo de iteraciones
# werror: Solución + array de errores
tol, Nmax = 1e-3, 50

sbis, errbis = bisection(f, 4.7, 5.2, tol, Nmax)        # Bisección
sfp, errfp = false_position(f, 4.7, 5.2, tol, Nmax)     # Falsa posición
spf, errpf = fixed_point(g, 4.7, tol, Nmax)             # Punto Fijo
snr, errnr = newton_raphson(f, df, 4.7, tol, Nmax)      # Newton-Raphson
ssec, errsec = secant(f, 4.7, 4.8, tol, Nmax)           # Secante

h, c, kb = 6.626e-34, 3e8, 1.381e-23

# Cálculo de x con precisión % 1e-3 (tal que Delta(b) = 1e-6)
print('Soluciones para f(x) = 0 con diferentes métodos')
print(f'Bisección: {sbis: .3f}')
print(f'Falsa posición: {sfp: .3f}')
print(f'Punto Fijo: {spf: .3f}')
print(f'Newton-Raphson: {snr: .3f}')
print(f'Secante: {ssec: .3f}')

# Cálculo de la constante b con precisión 1e-6
print('Constante b con diferentes métodos') # b = hc/(kb*x)
print(f'Bisección: {(h*c)/(kb*sbis): .6f}')
print(f'Falsa posición: {(h*c)/(kb*sfp): .6f}')
print(f'Punto Fijo: {(h*c)/(kb*spf): .6f}')
print(f'Newton-Raphson: {(h*c)/(kb*snr): .6f}')
print(f'Secante: {(h*c)/(kb*ssec): .6f}')

#### PLOT DE ERRORES POR ITERACIÓN ####
# len(error)+1: Número de iteraciones hasta obtener la precisión deseada
plt.plot(1+arange(len(errbis)), errbis, label='Bisección')
plt.plot(1+arange(len(errfp)), errfp, label='Falsa posición')
plt.plot(1+arange(len(errpf)), errpf, label='Punto Fijo')
plt.plot(1+arange(len(errnr)), errnr, label='Newton-Raphson')
plt.plot(1+arange(len(errsec)), errsec, label='Secante')
plt.title('Errores Relativos en cada Iteración')
plt.xlabel('Iteraciones')
plt.ylabel('Errores Relativos [%]')
plt.yscale('log')
plt.legend()
plt.show()