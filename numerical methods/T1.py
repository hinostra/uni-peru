#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

''' Mapa logístico de Feigenbaum '''

# RETORNA UN ARRAY CON LA SUCESIÓN x(n+1) = r*x(n)*(1-x(n))
# x0: Valor inicial
# n: Número de iteraciones
# r: constante
def feigenbaum(x0, n, r=1):
    x = zeros(n+1, float)
    x[0] = x0
    for k in range(n):
        x[k+1] = r*x[k]*(1-x[k])
    return x

# Primer ejercicio ##

Na = 1000                           # Número de iteraciones
a0 = [0.45, 0.5, 0.55]              # Valores iniciales
a = zeros([len(a0), Na+1], float)   # Array para x[k]

for i in range(len(a0)):
    a[i,:] = feigenbaum(a0[i], Na)

Ia = arange(Na+1)                   # Array de las iteraciones

for i in range(len(a0)):
    plt.scatter(Ia, a[i,:], label=f'$x_0={a0[i]}$', s=2)
    plt.plot(Ia, a[i, :], '--', lw=1)
plt.xlabel('Iteraciones')
plt.ylabel(r'$x_{k+1}$')
plt.title('Mapa Logístico de Feigenbaum')
plt.legend()
#plt.savefig('tarea1a.png', dpi=200)
plt.show()

## Segundo ejercicio ##

Nb = 3000                                 # Iteraciones
b0 = 0.5                                  # Valor inicial
rb = linspace(1, 4, 31)                   # Valores de r
b = zeros([size(rb), 2001], float)        # Array para x[k] en función de r

for i in range(len(rb)):
    b[i, :] = feigenbaum(b0, Nb, rb[i])[1000:]

for i, r in enumerate(rb):
    plt.scatter(r*ones(2001), b[i,:], s=2, alpha=0.5, c='purple')
plt.xlabel('r')
plt.ylabel(r'$x_{k+1}$')
plt.title('Mapa Logístico de Feigenbaum')
#plt.savefig('tarea1b.png', dpi=200)
plt.show()


''' El conjunto de Mandelbrot ''' 

# RETORNA z[n] O z[k] (k<n) si |z[k]| > 2
# c: constante
# n: número de iteraciones
# z: valor inicial
def mandelbrot(c, n=100, z=0):
    i = 0
    while i < n and abs(z) < 2:
        z = z**2 + c
        i += 1
    return z

## Versión 1 ##

# N = 100                                 # Número de puntos en la malla
# yy, xx = mgrid[2:-2:N*1j, -2:2:N*1j]    # Arrays de pares ordenados  
# cc = xx + yy*1j
# ms = zeros([N, N], int)

# for i in range(N):
#     for j in range(N):
#         z = mandelbrot(cc[i,j])
#         if abs(z) > 2:
#             ms[i,j] = 1

# plt.imshow(ms, cmap='gray', extent=(-2,2,-2,2))
# plt.xlabel('real(c)')
# plt.ylabel('imag(c)')
# plt.title('Conjunto de Mandelbrot')
# #plt.savefig('tarea1c.png', dpi=200)
# plt.show()

## Versión 2 ##

N = 100                           # Número de puntos en la malla
x = linspace(-2, 2, N)            # Coordenadas x
y = linspace(2, -2, N)            # Coordenadas y
ms = zeros([N, N], int)           # 0: pertenece, 1: no pertenece

for i in range(N):
    for j in range(N):
        z = mandelbrot(x[i] + 1j*y[j])
        if abs(z) > 2:
            ms[j, i] = 1

plt.imshow(ms, cmap='gray', extent=(-2,2,-2,2))
plt.xlabel('real(c)')
plt.ylabel('imag(c)')
plt.title('Conjunto de Mandelbrot')
#plt.savefig('tarea1c.png', dpi=200)
plt.show()
