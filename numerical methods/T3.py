#! /usr/bin/python3
from random import gauss
from telnetlib import GA
from numpy import *
import matplotlib.pyplot as plt
from modules.linear_systems import *

import warnings
warnings.filterwarnings("ignore")

#### EJERCICIO 1 ####
## Matrices triangular superior e inferior ##

# A1a = array([[1, 2, 1], [0, -4, 1], [0, 0, -2]])
# b1a = array([5, 2, 4])

# x1a = backward_substitution(A1a, b1a, 3)
# print('Solución del primer sistema ', x1a)

# A1b = array([[2, 0, 0], [1, 4, 0], [4, 3, 3]])
# b1b = array([4, 2, 5])

# x1b = forward_substitution(A1b, b1b, 3)
# print('Solución del segundo sistema ', x1b)


#### EJERCICIO 2 ####
## Sistema de resistores ##

def generator(N):
    A = zeros([N, N], float)
    if N == 2:
        A[0, 0], A[1, 1] = 3, 3
        A[0, 1], A[1, 0] = -1, -1
    else:
        for i in range(N-2):
            if i == 0:
                A[i, i] = 3
            else: 
                A[i, i] = 4
            A[i+1, i] = -1
            A[i+2, i] = -1 
            A[i, i+1] = -1
            A[i, i+2] = -1
        A[N-2, N-2], A[N-1, N-1] = 4, 3
        A[N-1, N-2], A[N-2, N-1] = -1, -1
    return A

# Vm = 5                              # V+
# A2a = generator(100)                   # Caso N=6
# b2a = zeros(100, float)
# b2a[0], b2a[1] = Vm, Vm
# x2a = gauss_elimination(A2a, b2a, 100)
# x2b, e = gauss_seidel(A2a, b2a, 100, emax=1e-6, nmax=2000)
# print('Para N=100 la solución es ', x2a)
# print('Para N=100 la solución es ', x2b)
# print(size(e))

# A2b = generator(1000)                   # Caso N=1000
# b2b = zeros(1000, float)
# b2b[0], b2b[1] = Vm, Vm
# x2b = gauss_elimination(A2b, b2b)
# print('Para N=1000 la solución es ', x2b)


#### EJERCICIO 3 ####
## Sistema de osciladores ##

# k = 10
# m1, m2, m3 = 1, 1, 1
# g = 10

# A3 = array([[3*k, -2*k, 0], [-2*k, 3*k, -k], [0, -k, k]], dtype=float)
# b3 = array([m1*g, m2*g, m3*g], dtype=float)

# x3ge = gauss_elimination(A3, b3, 3)
# LA3, UA3 = LU_decomposition(A3, 3)
# x3lu = LU_solution(LA3, UA3, b3, 3)
# print('Soluciones estacionarias del sistema de osciladores')
# print('Eliminación gaussiana: ', x3ge)
# print('Descomposición LU: ', x3lu)

# Prueba LU ##

# A = array([[3,1,3,3], [6,4,7,9], [6,6,10,13], [6,6,12,17]], float)
# l,u = LU_decomposition(A, 4)
# print(l)
# print(u)
# print(dot(l,u))

# # Prueba Cholesky ##

# A = array([[1,2,3,4],[2,8,12,16],[3,12,27,36],[4,16,36,64]], float)
# g = cholesky_decomposition(A, 4)
# print(g)
# print(dot(transpose(g),g))

# Prueba Métodos Iterativos
A = array([[4, 2, 3], [3, -5, 2], [-2, 3, 8]], float)
b = array([8, -14, 27], float)

xjacobi, ejacobi = jacobi(A, b, 3)
xgauss_seidel, egauss_seidel = gauss_seidel(A, b, 3)
xsor, esor = sor(A, b, 0.8, 3)
xexacta = gauss_elimination(A, b, 3)
print('Solución exacta: ', xexacta)
print(xjacobi, f' con {len(ejacobi)} iteraciones')
print(xgauss_seidel, f' con {len(egauss_seidel)} iteraciones')
print(xsor, f' con {len(esor)} iteraciones')