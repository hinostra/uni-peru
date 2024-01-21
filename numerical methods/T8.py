#! /usr/bin/python3
from cProfile import label
from numpy import *
import matplotlib.pyplot as plt
from modules.edo import *

import warnings
warnings.filterwarnings("ignore")

### TEST ####
# def feuler(x, y):
#     return -2*x**3 + 12*x**2 - 20*x + 8.5

# sol = euler(feuler, [0., 4.], 0.5, [1.])
# print(f'x = {sol[0,:]}')
# print('Método de Euler')
# print(f'y = {sol[1,:]}')
# sol = RK5(feuler, [0., 4.], 0.5, [1.])
# print('RK5')
# print(f'y = {sol[1,:]}')

#### EJERCICIO 1 ####
def f1(t, y):
    return y*t**3 - 1.5*y

def exact1(t):
    return exp(1/4*t**4 - 1.5*t)

tlims = [0., 2.]
h = 0.2
t = arange(0, 2.+h, h)
y0 = [1.]

yexacto = exact1(t)
soleuler = euler(f1, tlims, h, y0)
solheun = heun_simple(f1, tlims, h, y0)
solralston = RK2(f1, tlims, h, y0, 'Ralston')
solrk4 = RK4(f1, tlims, h, y0)
solheunm = heun_modificado(f1, tlims, h, y0)
# soladam = adam_bashforth(f1, tlims, h, y0, 3)

plt.plot(t, yexacto, label='Solución analítica')
plt.plot(soleuler[0,:], soleuler[1,:], label='Euler')
plt.plot(solheun[0,:], solheun[1,:], label='Heun')
plt.plot(solralston[0,:], solralston[1,:], label='Ralston')
plt.plot(solrk4[0,:], solrk4[1,:], label='RK4')
plt.plot(solheunm[0,:], solheunm[1,:], label='Heun modificado')
# plt.plot(soladam[0,:], soladam[1,:], label='Adam-Bashforth')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.show()

#### EJERCICIO 2 ####
