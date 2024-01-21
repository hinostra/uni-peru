#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt
from modules.polynomial_interpolation import *

import warnings
warnings.filterwarnings("ignore")

# #### TEST ####
# ttest = array([2, 2, 2], dtype=float)
# ftest = array([1, 2, 3], dtype=float)

# # Monomios
# lmtest = monomial_coeff(ttest, ftest, 3)
# print('Coeficientes de monomios: ', lmtest)
# print('Interpolación con monomios f(t=1.5) = ', monomial_interpolation(1.5, lmtest, 3))
# print(f'{lmtest[0]} + {lmtest[1]}*t + {lmtest[2]}*t^2')

# print('Interpolación con Lagrange f(t=1.5) = ', lagrange_interpolation(1.5, ttest, ftest, 3))

# lntest = newton_coeff(ttest, ftest, 3)
# print('Coeficientes de polinomios de Newton: ', lntest)
# print('Interpolación con Newton f(t=1.5) = ', newton_interpolation(1.5, ttest, lntest, 3))

# print('6 puntos de Chebyshev en [-1, 1]: ', chebyshev_points(-1., 1., 6))

# spline2test = quadratic_spline_coeff(ttest, ftest, 2)
# spline3test = cubic_spline_coeff(ttest, ftest, 2)

#### EJERCICIO 1 ####
x1 = array([0, 1, 2, 5.5, 11, 13, 16, 18], dtype=float)
y1 = array([0.5, 3.134, 5.3, 9.9, 10.2, 9.35, 7.2, 6.2], dtype=float)

# Coeficientes lambda para monomios y P. Newton
lm1 = monomial_coeff(x1, y1, len(x1))
ln1 = newton_coeff(x1, y1, len(x1))

# Interpolación
print('1. Interpolación para x = 8:')
print('\tMonomios: y(8) = ', monomial_interpolation(8., lm1, len(x1)))
print('\tLagrange: y(8) = ', lagrange_interpolation(8., x1, y1, len(x1)))
print('\tNewton: y(8) = ', newton_interpolation(8., x1, ln1, len(x1)))

#### EJERCICIO 2 ####
def f(x):
    return tanh(20*sin(12*x)) + 1/50*exp(3*x)*sin(300*x)

# 100 Puntos equidistantes
x2e100 = linspace(0, 1, 100)
y2e100 = f(x2e100)
# 100 Puntos de Chebyshev
x2c100 = chebyshev_points(0, 1, 100)
y2c100 = f(x2c100)

# Interpolación con polinomios de Lagrange
x2100 = linspace(0, 1, 250)
i2e100 = lagrange_interpolation(x2100, x2e100, y2e100, 100)
i2c100 = lagrange_interpolation(x2100, x2c100, y2c100, 100)

plt.plot(x2100, abs(f(x2100) - i2e100), color='blue', label='Puntos equidistantes')
plt.plot(x2100, abs(f(x2100) - i2c100), color='red', label='Puntos de Chebyshev')
plt.title('Error para puntos equidistantes y de Chebyshev')
plt.xlabel('x')
plt.ylabel(r'$|f(x) - \Pi f(x)|$')
plt.yscale('log')
plt.legend()
plt.show()

error2 = list()
x2aux = linspace(0, 1, 2000)
for n in range(1, 11):
    x2caux = chebyshev_points(0, 1, n*100)
    y2caux = f(x2caux)

    i2aux = lagrange_interpolation(x2aux, x2caux, y2caux, n*100)
    err2 = max(abs(f(x2aux) - i2aux))
    error2.append(err2)

plt.plot([i*100 for i in range(1, 11)], error2)
plt.xlabel('n')
plt.ylabel(r'max($|f(x) - \Pi f(x)|$)')
plt.title('Puntos de Chebyshev para diferentes n')
plt.yscale('log')
plt.show()

#### EJERCICIO 3 ####
def g(t):
    return (sin(t))**2

a, b = 0, 2*pi
n3e, n3c = 4, 4
x3e = linspace(a, b, n3e)
y3e = g(x3e)
x3c = chebyshev_points(a, b, n3c)
y3c = g(x3c)

x3 = linspace(a, b, 4)
i3e = lagrange_interpolation(x3, x3e, y3e,n3e)
i3c = lagrange_interpolation(x3, x3c, y3c, n3c)
abcd3e = cubic_spline_coeff(x3e, y3e, n3e-1)
print(y3e)
sc3e = array([cubic_spline(x, x3e, abcd3e, n3e-1) for x in x3]) 

plt.plot(x3, g(x3), label=r'$sin^2(t)$')
#plt.plot(x3, i3e, label='Polinomio (Equidistantes) n=7')
plt.plot(x3, i3c, label='Polinomio (Chebyshev) n=7')
plt.plot(x3, sc3e, label='Spline cúbica')
plt.title('Puntos equidistantes')
plt.xlabel('t')
plt.legend()
plt.show()