#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt
from modules.edp import *

import warnings
warnings.filterwarnings("ignore")

# Ejercicio 1: Ecuación de Laplace
def f1(x,y):
    return 0

alpha, beta = 0, 0
neumann = [alpha, beta]
uleft = array([0, 25, 50, 75, 100], float)
utop = array([0, 25, 50, 75, 100])
dirichlet = [uleft, utop]
lims = [0,10]

U = Poisson_ND(f1, neumann, dirichlet, lims)

Uplot = plt.imshow(transpose(U), origin='lower')
cbar = plt.colorbar(Uplot)
cbar.set_label(r'T(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ejercicio 1')
plt.show()

# Ejercicio 2: Ecuación parabólica

dx, J = 0.25, 4
dt, N = 0.01, 10
dirichlet = ones(N+1)
cond_ini = zeros(J+1)

solFTCS = Parabolica_FTCS(dirichlet, cond_ini, dx, dt, N, J)
solDF = Parabolica_Dufort_Frankel(dirichlet, cond_ini, dx, dt, N, J)
solIm = Parabolica_Implicito(dirichlet, cond_ini, dx, dt, N, J)
solCN = Parabolica_Crank_Nicolson(dirichlet, cond_ini, dx, dt, N, J)

x = linspace(0, 1, J+1)
t = linspace(0, 0.1, N+1)

for i in range(N+1):
    plt.plot(x, solFTCS[i,:], label=t[i])
plt.legend()
plt.xlabel('x')
plt.ylabel('U(x,t)')
plt.title('2. Solución por FTCS')
plt.show()

for i in range(N+1):
    plt.plot(x, solDF[i,:], label=t[i])
plt.legend()
plt.xlabel('x')
plt.ylabel('U(x,t)')
plt.title('2. Solución por Dufort-Frankel')
plt.show()

for i in range(N+1):
    plt.plot(x, solIm[i,:], label=t[i])
plt.legend()
plt.xlabel('x')
plt.ylabel('U(x,t)')
plt.title('2. Solución por Método completamente implícito')
plt.show()

for i in range(N+1):
    plt.plot(x, solCN[i,:], label=t[i])
plt.legend()
plt.xlabel('x')
plt.ylabel('U(x,t)')
plt.title('2. Solución por Crank-Nicholson')
plt.show()