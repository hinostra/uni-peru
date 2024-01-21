#! /usr/bin/python3
from cProfile import label
from numpy import *
import matplotlib.pyplot as plt
from modules.curve_fitting import *

import warnings
warnings.filterwarnings("ignore")

# # Test de linear regression
# xt1 = linspace(1, 7, 7)
# yt1 = array([0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5])
# at1, _ = linear_regression(xt1, yt1, 7)
# print(at1)

# xt2 = linspace(0, 5, 6)
# yt2 = array([2.1, 7.7, 13.6, 27.2, 40.9, 61.1])
# at2, _ = polinomial_regression(xt2, yt2, 6, 2)
# print(at2)

# xt3 = array([0.25, 0.75, 1.25, 1.75, 2.25])
# yt3 = array([0.28, 0.57, 0.68, 0.74, 0.79])

# def f(x, b0, b1):
#     return b0*(1 - exp(-b1*x))

# def df1(x, b0, b1):
#     return 1 - exp(-b1*x)

# def df2(x, b0, b1):
#     return b0*x*exp(-b1*x)

# df = [df1, df2]
# a0 = array([1., 1.])

# print(nonlinear_regression(f, df, xt3, yt3, a0, 5, 2, nmax=1))

#### EJERCICIO 1 ####

# Datos
x1 = array([0.5, 1., 2., 3., 4.])
y1 = array([10.4, 5.8, 3.3, 2.4, 2.])

# v1 = 1/b + a/b*u1
u1 = 1/sqrt(x1)
v1 = sqrt(y1)

a1, _ = linear_regression(u1, v1, 5)
print('1. Tras linealizar la ecuación se obtiene')
print(f'a = {a1[1]/a1[0]}, b = {1/a1[0]}')

def f1(x, a, b):
    return (a + sqrt(x))**2/(b*sqrt(x))**2

x1plot = linspace(0.5, 4, 50)
y1plot = f1(x1plot, a1[1]/a1[0], 1/a1[0])

plt.plot(x1plot, y1plot, color='k', label='Ajuste')
plt.scatter(x1, y1, s=7, label='Datos')
plt.legend()
plt.show()

#### EJERCICIO 2 ####
# y = a*x*exp(b*x)
x2 = array([0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.5, 1.7, 1.8])
y2 = array([0.75, 1.25, 1.45, 1.25, 0.85, 0.55, 0.35, 0.28, 0.18])

# v2 = ln(alpha) + beta*x
v2 = log(y2/x2)

a2l, _ = linear_regression(x2, v2, len(x2))
print('2. Tras linealizar la ecuación se obtiene')
print(f'beta = {a2l[1]}, alpha = {exp(a2l[0])}')

def f2(x, a, b):
    return a*x*exp(b*x)

def df21(x, a, b):
    return x*exp(b*x)

def df22(x, a, b):
    return a*x**2*exp(b*x)

a20 = array([9., -2.])
df2 = [df21, df22]

a2n = nonlinear_regression(f2, df2, x2, y2, a20, len(x2), 2, emax=1e-3, nmax=100)
print('2. Mediante una regresión no lineal')
print(f'beta = {a2n[1]}, alpha = {a2n[0]}')

x2plot = linspace(0.1, 1.8, 50)
y2plotl = f2(x2plot, exp(a2l[0]), a2l[1])
y2plotn = f2(x2plot, a2n[0], a2n[1])

plt.plot(x2plot, y2plotl, label='Lineal')
plt.plot(x2plot, y2plotn, label='No lineal')
plt.scatter(x2, y2)
plt.legend()
plt.show()