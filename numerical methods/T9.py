#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt
from modules.boundary_problems import *
from modules.eigenvalue_problems import *

import warnings
warnings.filterwarnings("ignore")


#### EJERCICIO 1 ####
L = 10.
h = 0.01
T0, TL = 40, 200
Ta = 20
Tcond = [T0, TL]
dx = 1.
N = int(L/dx) + 1

def exact1(x):
    num = (TL-Ta)*sinh(sqrt(h)*x) + (T0-Ta)*sinh(sqrt(h)*(L-x))
    den = sinh(sqrt(h)*L)
    return Ta + num/den

def f1(x):
    return -h*Ta

sol_num = D2_U_dirichlet(f1, [0, L], dx, -h, Tcond)
xtest = linspace(0, L, N)
sol_ex = exact1(xtest)

plt.plot(xtest, sol_ex, label='Exacta', marker='s')
plt.plot(sol_num[0,:], sol_num[1,:], label='Numérica', marker='o')
plt.title('Ejercicio 1. Solución de D^2 T + h(Ta-T) = 0')
plt.xlabel('x')
plt.ylabel('T')
plt.legend()
plt.show()

error = list()
dx_array = [0.5, 1., 2., 2.5]
for dx in dx_array:
    numerical = D2_U_dirichlet(f1, [0, L], dx, -h, Tcond)
    x1 = numerical[0,:]
    T1 = numerical[1,:]
    er = max(abs(T1 - exact1(x1)))
    error.append(er)


plt.plot(dx_array, error)
plt.yscale('log')
plt.xscale('log')
plt.title('Ejercicio 1. Errores de refinamiento')
plt.xlabel(r'$\Delta x$')
plt.ylabel('Error')
plt.show()


#### EJERCICIO 2 ####

L = 10.
h = 0.01
dx = 1.
Tinf = 40
TL = 200
dTa = 10
Tcond = [dTa, TL]
N = int(L/dx) + 1

def exact2(x):
    C1 = (TL - Tinf)/cosh(sqrt(h)*L)
    C2 = dTa/(sqrt(h)*cosh(sqrt(h)*L))
    return Tinf + C1*cosh(sqrt(h)*x) + C2*sinh(sqrt(h)*(x-L))

def f2(x):
    return -h*Tinf

sol_num = D2_U_neumann(f2, [0, L], dx, -h, Tcond)
xtest = linspace(0, L, N)
sol_ex = exact2(xtest)

plt.plot(xtest, sol_ex, label='Exacta', marker='s')
plt.plot(sol_num[0,:], sol_num[1,:], label='Numérica', marker='o')
plt.title(r'Ejercicio 2. Solución de D^2 T + h(T$_\infty$-T) = 0')
plt.xlabel('x')
plt.ylabel('T')
plt.legend()
plt.show()

error = list()
dx_array = [2.5, 2., 1., 0.5, 0.1]
for dx in dx_array:
    numerical = D2_U_neumann(f2, [0, L], dx, -h, Tcond)
    x2 = numerical[0,:]
    T2 = numerical[1,:]
    er = max(abs(T2 - exact2(x2)))
    error.append(er)

plt.plot(dx_array, error)
plt.yscale('log')
plt.xscale('log')
plt.title('Ejercicio 2. Errores de refinamiento')
plt.xlabel(r'$\Delta x$')
plt.ylabel('Error')
plt.show()


# #### EJERCICIO 3 ####

# A41 = array([[-4., 10.], [7., 5.]])
# A42 = array([[1., 2., -2.], [-2., 5., -2.], [-6., 6., -3.]])
# x01 = array([1., 0.])
# x02 = array([1., 0., 0.])

# eigen41 = potencias(A41, x01, nmax=20)
# eigen42 = potencias(A42, x02, nmax=20)

# print('Autovalor de la primera matriz: ', eigen41)
# print('Autovalor de la segunda matriz: ', eigen42)