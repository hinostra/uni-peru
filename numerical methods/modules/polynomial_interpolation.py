#! /usr/bin/python3
from numpy import *
from linear_systems import *

import warnings
warnings.filterwarnings("ignore")

''' BASE DE MONOMIOS '''

## Coeficientes lambda: f[n-1](t) = lambda0*t^0 + lambda1*t^1 + ... + lambdan-1*t^(n-1)
# xdata: t[0], t[1], ..., t[n-1]
# ydata: y[0], y[1], ..., y[n-1]
# n: número de puntos
def monomial_coeff(xdata, ydata, n: int):
    A = ones([n, n])
    for i in range(n):
        for j in range(1, n):
            A[i,j] = xdata[i]**j
    
    lambda_coeff = gauss_elimination(A, ydata, n)
    return lambda_coeff

## Interpolación para un valor x
# x: variable
# coeff: coeficientes lambda
# n: número de puntos
def monomial_interpolation(x, coeff, n: int):
    fx = coeff[0]
    for i in range(1, n):
        fx += coeff[i]*x**i
    return fx


''' BASE DE POLINOMIOS DE LAGRANGE '''

## Polinomio de Lagrange para el índice k
# x: variable
# xdata: t[1], t[2], ...
# k: número de polinomio
# n: número de puntos
def lagrange_pols(x, xdata, k: int, n: int):
    lx = 1
    for i in range(n):
        if i != k:
            lx *= (x - xdata[i])/(xdata[k] - xdata[i])
    return lx

## Interpolación para un valor x
# x: variable
# xdata: t[1], t[2], ...
# ydata: y[1], y[2], ...
# n: número de puntos
def lagrange_interpolation(x, xdata, ydata, n: int):
    fx = 0
    for k in range(n):
        fx += ydata[k]*lagrange_pols(x, xdata, k, n)
    return fx


''' BASE DE POLINOMIOS DE NEWTON '''

## Polinomio de Newton para el índice k
# x: variable
# xdata: t[1], t[2], ...
# k: número de polinomio, k<n
def newton_pols(x, xdata, k: int):
    lx = 1
    if k >= 1:
        for i in range(k):
            lx *= (x - xdata[i])
    return lx

## Coeficientes lambda: lambda[0], lambda[1], ..., lambda[n-1]
# xdata: t[0], t[1], ..., t[n-1]
# ydata: y[0], y[1], ..., t[n-1]
# n: número de puntos
def newton_coeff(xdata, ydata, n: int):
    A = zeros([n, n])
    for i in range(n):
        for j in range(i+1):
            A[i, j] = newton_pols(xdata[i], xdata, j)

    lambda_coeff = forward_substitution(A, ydata, n)
    return lambda_coeff

## Interpolación para un valor x
# x: variable
# xdata: t[1], t[2], ...
# coeff: coeficientes lambda
# n: número de puntos
def newton_interpolation(x, xdata, coeff, n: int):
    fx = coeff[0]
    for i in range(1, n):
        fx += coeff[i]*newton_pols(x, xdata, i)
    return fx


''' PUNTOS DE CHEBYSHEV '''

## Puntos (extremales) de Chebyshev
# [a, b]: intervalo
# n: número de puntos
def chebyshev_points(a, b, n):
    x = zeros(n)
    for i in range(n):
        x[i] = (a+b)/2. + (a-b)/2.*cos(pi*float(i)/float(n-1))
    return x

''' SPLINE '''

## Spline lineal
# x: variable
# xdata: x[0], x[1], ..., x[n]
# fdata: f(x[0]), f(x[1]), ..., f(x[n])
# n+1: número de puntos
def linear_spline(x, xdata, fdata, n):
    fx = 0
    for i in range(n):
        if x > xdata[i] and x < xdata[i+1]:
            m = (fdata[i+1] - fdata[i])/(xdata[i+1] - xdata[i])
            fx = fdata[i] + m*(x - x[i])
    return fx

## Spline cuadrático

# Coeficientes
# xdata: x[0], x[1], ..., x[n]
# fdata: f(x[0]), f(x[1]), ..., f(x[n])
# n: número de splines (número de puntos -1)
def quadratic_spline_coeff(xdata, fdata, n):
    M = zeros([3*n, 3*n], dtype=float)
    Y = zeros(3*n, dtype=float)
    # Construcción de la matriz
    for i in range(n):
        M[2*i, i], M[2*i+1, i] = xdata[i]**2, xdata[i+1]**2
        M[2*i, n+i], M[2*i+1, n+i] = xdata[i], xdata[i+1]
        M[2*i, 2*n+i], M[2*i+1, 2*n+i] = 1, 1
        Y[2*i], Y[2*i+1] = fdata[i], fdata[i+1]
    for i in range(n-1):
        M[2*n+i, i], M[2*n+i, i+1] = 2*xdata[i+1], -2*xdata[i+1]
        M[2*n+i, n+i], M[2*n+i, n+i+1] = 1., -1.
    M[-1,0] = 2

    # Pivot si elementos de la diagonal son iguales a cero
    for i in range(3*n-1, -1, -1):
        if M[i, i] == 0:
            pivot(M, Y, i, False)

    # Solución con Eliminación Gaussiana si el pivot no ayudó
    if min(abs(M.diagonal())) == 0:
        X = gauss_elimination(M, Y, 3*n)
        print('Se tuvo que usar Eliminación Gaussiana porque')
        print(diag(M))
    else:
        X, _ = sor(M, Y, 0.9, 3*n, nmax=200)
    
    return X

# Spline
# x: variable
# xdata: x[0], x[1], ..., x[n]
# coeff: coeficientes a[i], b[i], c[i]
# n: número de splines (número de puntos -1)
def quadratic_spline(x, xdata, coeff, n):
    fx = 0
    for i in range(n):
        if x >= xdata[i] and x <= xdata[i+1]:
            fx = coeff[i]*x**2 + coeff[n+i]*x + coeff[2*n+i]
    return fx

## Spline cúbico

# Coeficientes
# xdata: x[0], x[1], ..., x[n]
# fdata: f(x[0]), f(x[1]), ..., f(x[n])
# n: número de splines (número de puntos -1)
def cubic_spline_coeff(xdata, fdata, n):
    M = zeros([4*n, 4*n], dtype=float)
    Y = zeros(4*n, dtype=float)
    # Construcción de la matriz
    for i in range(n):
        M[2*i, i], M[2*i+1, i] = xdata[i]**3, xdata[i+1]**3
        M[2*i, n+i], M[2*i+1, n+i] = xdata[i]**2, xdata[i+1]**2
        M[2*i, 2*n+i], M[2*i+1, 2*n+i] = xdata[i], xdata[i+1]
        M[2*i, 3*n+i], M[2*i+1, 3*n+i] = 1, 1
        Y[2*i], Y[2*i+1] = fdata[i], fdata[i+1]
    for i in range(n-1):
        M[2*n+i, i], M[2*n+i, i+1] = 3*xdata[i+1]**2, -3*xdata[i+1]**2
        M[2*n+i, n+i], M[2*n+i, n+i+1] = 2*xdata[i+1], -2*xdata[i+1]
        M[2*n+i, 2*n+i], M[2*n+i, 2*n+i+1] = 1., -1.
    for i in range(n-1):
        M[3*n-1+i, i], M[3*n-1+i, i+1] = 6*xdata[i+1], -6*xdata[i+1]
        M[3*n-1+i, n+i], M[3*n-1+i, n+i+1] = 2., -2.
    M[-2,0], M[-2,n] = 6*xdata[0], 2.
    M[-1,n-1], M[-1,2*n-1] = 6*xdata[n], 2.

    # Pivot si elementos de la diagonal son iguales a cero
    for i in range(4*n-1, -1, -1):
        if M[i, i] == 0:
            pivot(M, Y, i, False)

    # # Solución con Eliminación Gaussiana si el pivot no ayudó
    # if min(abs(M.diagonal())) == 0:
    #     X = gauss_elimination(M, Y, 4*n)
    #     #print('Se tuvo que usar Eliminación Gaussiana porque')
    #     #print(diag(M))
    # else:
    #     X, _ = sor(M, Y, 0.9, 4*n, nmax=200)
    X = gauss_elimination(M, Y, 4*n)
    print(X)
    return X

# Spline
# x: variable
# xdata: x[0], x[1], ..., x[n]
# coeff: coeficientes a[i], b[i], c[i]
# n: número de splines (número de puntos -1)
def cubic_spline(x, xdata, coeff, n):
    fx = 0
    for i in range(n):
        if x >= xdata[i] and x <= xdata[i+1]:
            fx = coeff[i]*x**3 + coeff[n+i]*x**2 + coeff[2*n+i]*x + coeff[3*n+i]
    return fx