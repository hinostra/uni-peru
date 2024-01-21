#! /usr/bin/python3
from random import gauss
from numpy import *
import matplotlib.pyplot as plt
from linear_systems import gauss_seidel

import warnings
warnings.filterwarnings("ignore")


''' ECUACIÓN DE POISSON '''
# Laplaciano(u) = f(x)
C = array([[-4, 2, 0, 0], [1, -4, 1, 0], [0, 1, -4, 1], [0, 0, 1, -4]], float)

# Falta solo con condiciones de Dirichlet

# Condiciones de Neumann y Dirichlet
# Malla cuadrada (índices de izq a der, de abajo a arriba)
def Poisson_ND(f, neumann, dirichlet, lims=[0.,1.]):
    alpha, beta = neumann       # Condiciones de Neumann: DxU = alpha, DyU = beta
    uleft, utop = dirichlet     # Condiciones de Dirichlet: U[4,j], U[i,4]
    n = 5                       # Número de puntos en la malla
    x = linspace(lims[0], lims[1], n)
    h = (lims[1]-lims[0])/n
    U = zeros([n ,n], float)    # Solución U[i,j] = U(x[i], y[j])
    U[n-1,:] = uleft
    U[:,n-1] = utop

    # Matriz del sistema lineal
    M = zeros([16, 16], float)
    M[0:4,0:4] = copy(C)
    M[4:8,4:8] = copy(C)
    M[8:12,8:12] = copy(C)
    M[12:16,12:16] = copy(C)
    M[0:4,4:8] = 2*copy(eye(4))
    M[4:8,0:4] = copy(eye(4))
    M[4:8,8:12] = copy(eye(4))
    M[8:12,4:8] = copy(eye(4))
    M[8:12,12:16] = copy(eye(4))
    M[12:16,8:12] = copy(eye(4))

    F = zeros(16, float)
    F[0] = f(x[0],x[0]) + 2*alpha/h + 2*beta/h
    F[1] = f(x[1],x[0]) + 2*beta/h 
    F[2] = f(x[2],x[0]) + 2*beta/h 
    F[3] = f(x[3],x[0]) + 2*beta/h
    F[4] = f(x[0],x[1]) + 2*alpha/h
    F[5] = f(x[1],x[1])
    F[6] = f(x[2],x[1])
    F[7] = f(x[3],x[1]) - U[4,1]/h**2
    F[8] = f(x[0],x[2]) + 2*alpha/h
    F[9] = f(x[1],x[2])
    F[10] = f(x[2],x[2])
    F[11] = f(x[3],x[2]) - U[4,2]/h**2
    F[12] = f(x[0],x[3]) + 2*alpha/h - U[0,4]/h**2
    F[13] = f(x[3],x[2]) - U[1,4]/h**2
    F[14] = f(x[3],x[2]) - U[2,4]/h**2
    F[15] = f(x[3],x[2]) - (U[3,4] + U[4,3])/h**2

    x, _ = gauss_seidel(M, F, 16)
    U[:4,0] = x[0:4]
    U[:4,1] = x[4:8]
    U[:4,2] = x[8:12]
    U[:4,3] = x[12:16]
    return U

# Falta la de calor

# N: número de puntos en t
# J: número de puntos en x
def Parabolica_FTCS(dirichlet, cond_ini, dx, dt, N, J):
    U = zeros([N+1, J+1], float)
    U[0,:] = cond_ini
    U[:,J] = dirichlet
    s = dt/dx**2
    for n in range(N):
        U[n+1,0] = (1-2*s)*U[n,0] + 2*s*U[n,1]
        for j in range(1,J):
            U[n+1,j] = s*(1-dx/2)*U[n,j-1] - (1-2*s)*U[n,j] + s*(1+dx/2)*U[n,j+1]
    return U

def Parabolica_Dufort_Frankel(dirichlet, cond_ini, dx, dt, N, J):
    U = zeros([N+1, J+1], float)
    U[:,J] = dirichlet
    s = dt/dx**2
    # Primer punto del método FTCS
    U[0:2,:] = Parabolica_FTCS(dirichlet[0:2], cond_ini, dx, dt, 1, J)

    for n in range(1,N):
        U[n+1,0] = (1-2*s)/(1+2*s)*U[n-1,0] + 4*s/(1+2*s)*U[n,1]
        for j in range(1,J):
            U[n+1,j] = (1-2*s)/(1+2*s)*U[n-1,j] + 2*s/(1+2*s)*(1-dx/2)*U[n,j-1] + 2*s/(1+2*s)*(1+dx/2)*U[n,j+1]
    return U

def Parabolica_Implicito(dirichlet, cond_ini, dx, dt, N, J):
    U = zeros([N+1, J+1], float)
    U[0,:] = cond_ini
    U[:,J] = dirichlet
    s = dt/dx**2

    for n in range(N):
        M = zeros([J, J], float)
        F = zeros(J, float)

        M[0,0], M[0,1] = (1+2*s), -2*s
        F[0] = U[n,0]
        for j in range(1,J-1):
            M[j,j-1] = -s*(1-dx/2)
            M[j,j] = 1+2*s
            M[j,j+1] = -s*(1+dx/2)
            F[j] = U[n,j]
        M[J-1,J-2], M[J-1,J-1] = -s*(1-dx/2), 1+2*s
        F[J-1] = U[n,J-1] + U[n+1,J]*s*(1+dx/2)

        U[n+1,:J], _ = gauss_seidel(M, F, J)
    return U

def Parabolica_Crank_Nicolson(dirichlet, cond_ini, dx, dt, N, J):
    U = zeros([N+1, J+1], float)
    U[0,:] = cond_ini
    U[:,J] = dirichlet
    s = dt/dx**2

    for n in range(N):
        M = zeros([J, J], float)
        F = zeros(J, float)

        M[0,0], M[0,1] = 1+s, -s
        F[0] = (1-s)*U[n,0] + s*U[n,1]
        for j in range(1,J-1):
            M[j,j-1] = -s/2*(1-dx/2)
            M[j,j] = 1+s
            M[j,j+1] = -s/2*(1+dx/2)
            F[j] = s/2*(1-dx/2)*U[n,j-1] + (1-s)*U[n,j] + s/2*(1+dx/2)*U[n,j+1]
        M[J-1,J-2], M[J-1,J-1] = -s/2*(1-dx/2), 1+s
        F[J-1] = s/2*(1-dx/2)*U[n,J-2] + (1-s)*U[n,J-1] + s/2*(1+dx/2)*U[n,J] + s/2*(1+dx/2)*U[n+1,J]

        U[n+1,:J], _ = gauss_seidel(M, F, J)
    return U
