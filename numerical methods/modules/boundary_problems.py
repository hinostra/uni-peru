#! /usr/bin/python3
from numpy import *
from linear_systems import gradiente_conjugado

import warnings
warnings.filterwarnings("ignore")

def D2_dirichlet(f, xlims, h, ucond, emax=1e-4):
    x0, xn = xlims
    n = int(abs(xn-x0)/h)
    u0, un = ucond

    sol = zeros([2, n+1], float)
    sol[0,:] = linspace(x0, xn, n+1)
    sol[1,0], sol[1,n] = u0, un

    A = zeros([n-1,n-1],float)
    A[0,0], A[0,1] = -2/h**2, 1/h**2
    A[n-2,n-3], A[n-2, n-2] = 1/h**2, -2/h**2

    F = zeros(n-1, float)
    F[0] = f(sol[0,1]) - u0/h**2
    F[n-2] = f(sol[0,n-1]) - un/h**2

    for i in range(1,n-2):
        A[i,i-1] = 1/h**2
        A[i,i] = -2/h**2
        A[i,i+1] = 1/h**2
        F[i] = f(sol[0,i+1])

    sol[1,1:n], _ = gradiente_conjugado(A, F, n-1, emax)
    return sol

def D2_neumann(f, xlims, h, ucond, emax=1e-4):
    x0, xn = xlims
    n = int(abs(xn-x0)/h)
    alpha, un = ucond

    sol = zeros([2, n+1], float)
    sol[0,:] = linspace(x0, xn, n+1)
    sol[1,n] = un

    A = zeros([n,n],float)
    A[0,0], A[0,1] = -1/h**2, 1/h**2
    A[n-1,n-2], A[n-1, n-1] = 1/h**2, -2/h**2

    F = zeros(n, float)
    F[0] = f(sol[0,0])/2 + alpha/h
    F[n-1] = f(sol[0,n-1]) - un/h**2

    for i in range(1,n-1):
        A[i,i-1] = 1/h**2
        A[i,i] = -2/h**2
        A[i,i+1] = 1/h**2
        F[i] = f(sol[0,i])

    sol[1,0:n], _ = gradiente_conjugado(A, F, n, emax)
    return sol

def D2_U_dirichlet(f, xlims, h, a, ucond, emax=1e-4):
    x0, xn = xlims
    n = int(abs(xn-x0)/h)
    u0, un = ucond

    sol = zeros([2, n+1], float)
    sol[0,:] = linspace(x0, xn, n+1)
    sol[1,0], sol[1,n] = u0, un

    A = zeros([n-1,n-1],float)
    A[0,0], A[0,1] = -2/h**2 + a, 1/h**2
    A[n-2,n-3], A[n-2, n-2] = 1/h**2, -2/h**2 + a

    F = zeros(n-1, float)
    F[0] = f(sol[0,1]) - u0/h**2
    F[n-2] = f(sol[0,n-1]) - un/h**2

    for i in range(1,n-2):
        A[i,i-1] = 1/h**2
        A[i,i] = -2/h**2 + a
        A[i,i+1] = 1/h**2
        F[i] = f(sol[0,i+1])

    sol[1,1:n], _ = gradiente_conjugado(A, F, n-1, emax)
    #sol[1,1:n], e = sor(A, F, 0.8, n-1, emax, 500)
    return sol

def D2_U_neumann(f, xlims, h, a, ucond, emax=1e-4):
    x0, xn = xlims
    n = int(abs(xn-x0)/h)
    alpha, un = ucond

    sol = zeros([2, n+1], float)
    sol[0,:] = linspace(x0, xn, n+1)
    sol[1,n] = un

    A = zeros([n,n],float)
    A[0,0], A[0,1] = -1/h**2 + a/2, 1/h**2
    A[n-1,n-2], A[n-1, n-1] = 1/h**2, -2/h**2 + a

    F = zeros(n, float)
    F[0] = f(sol[0,0])/2 + alpha/h
    F[n-1] = f(sol[0,n-1]) - un/h**2

    for i in range(1,n-1):
        A[i,i-1] = 1/h**2
        A[i,i] = -2/h**2 + a
        A[i,i+1] = 1/h**2
        F[i] = f(sol[0,i])

    sol[1,0:n], _ = gradiente_conjugado(A, F, n, emax)
    return sol