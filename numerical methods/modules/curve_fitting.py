#! /usr/bin/python3
from numpy import *
from linear_systems import gauss_elimination

import warnings
warnings.filterwarnings("ignore")

''' REGRESIÓN LINEAL '''
# xdata, ydata: mediciones (y = a0 + a1*x)
# n: número de datos
# return [parámetros], [r^2, std]
def linear_regression(xdata, ydata, n):
    sumx, sumy = 0, 0
    sumx2, sumxy = 0, 0
    st, sr = 0, 0

    # sum(xdata), sum(ydata), 
    # sum(xdata*ydata), sum(xdata**2)
    for i in range(n):
        sumx += xdata[i]
        sumy += ydata[i]
        sumxy += xdata[i]*ydata[i]
        sumx2 += xdata[i]**2

    # mean(x), mean(y)
    xm, ym = sumx/n, sumy/n

    # Cálculo de parámetros
    a1 = (n*sumxy - sumx*sumy)/(n*sumx2 - sumx**2)
    a0 = ym - a1*xm

    # St y sSr
    for i in range(n):
        st += (ydata[i] - ym)**2
        sr += (ydata[i] - a1*xdata[i] - a0)**2

    # Desviación estandar
    syx = sqrt(sr/(n-2))    # debido al ajuste

    # Coeficiente de determinación
    r2 = (st - sr)/st

    return array([a0, a1]), array([r2, syx])

''' REGRESIÓN POLINOMIAL '''
# xdata, ydata: mediciones (y = a0 + a1*x + ... + am*x^m)
# n: número de datos
# return [parámetros], [r^2, std]
def polinomial_regression(xdata, ydata, n, order):
    sr, st = 0, 0

    # Sistema lineal para los coeficientes
    A = zeros([order+1, order+1], float)
    b = zeros(order+1, float)
    for i in range(order+1):
        for j in range(i, order+1):
            k = i + j
            sum = 0
            for l in range(n):
                sum += xdata[l]**k
            A[i,j], A[j,i] = sum, sum
        sum = 0
        for l in range(n):
            sum += ydata[l]*xdata[l]**i
        b[i] = sum

    # Parámetros a[k]
    a = gauss_elimination(A, b, order+1)

    # St
    ym = mean(ydata)
    for i in range(n):
        st += (ydata[i] - ym)**2

    # Sr
    for i in range(n):
        aux = 0
        for j in range(order+1):
            aux += a[j]*xdata[i]**j
        sr += (ydata[i] - aux)**2

    # Error estandard de la estimación
    syx = sqrt(sr/(n-(order+1)))

    # Coeficiente de determinación
    r2 = (st - sr)/st

    return a, array([r2, syx])

''' REGRESIÓN LINEAL MÚLTIPLE '''
# xdata, ydata: mediciones (y = a0 + a1*x1 + ... + am*xm)
# n: número de datos
# return [parámetros], [r^2, std]
def multiple_linear_regression(xdata, ydata, n, m):
    sr, st = 0, 0

    # Sistema lineal para los coeficientes
    A = zeros([m+1, m+1], float)
    b = zeros(m+1, float)

    A[0, 0] = n
    for i in range(m):
        # Primera fila
        sum, aux = 0, 0
        for l in range(l):
            sum += xdata[i, l]
            aux += ydata[l]
        A[0, i+1], A[i+1, 0] = sum, sum
        b[0] = aux

        # Otras filas
        for j in range(i, m):
            sum = 0
            for l in range(n):
                sum += xdata[i, l]*xdata[j, l]
            A[i+1, j+1], A[j+1, i+1] = sum, sum
            
        # Vector b
        sum = 0
        for l in range(n):
            sum += xdata[i, l]*ydata[l]
        b[i+1] = sum

    # Parámetros a[k]
    a = gauss_elimination(A, b, m+1)

    # St
    ym = mean(ydata)
    for i in range(n):
        st += (ydata[i] - ym)**2

    # Sr
    for i in range(n):
        aux = 0
        for j in range(m):
            aux += a[j+1]*xdata[j, i]
        sr += (ydata[i] - a[0] - aux)**2

    # Error estandard de la estimación
    syx = sqrt(sr/(n-(m+1)))

    # Coeficiente de determinación
    r2 = (st - sr)/st

    return a, array([r2, syx])

''' REGRESIÓN NO LINEAL '''
# f: función no lineal
# df: lista de df/dak
# a0: estimación inicial de los parámetros
# n: número de puntos
# m: número de parámetros
def nonlinear_regression(f, df, xdata, ydata, a0, n, m, emax=1e-6, nmax=50):
    a, an = a0, a0
    D = zeros(n, float)
    Z = zeros([n, m], float)
    C = zeros([m, m], float)
    d = zeros(m, float)
    k = 0
    er = 2*emax
    while er > emax and k<nmax:
        a = an
        D = ydata - f(xdata, *a)
        for i in range(n):
            for j, g in enumerate(df):
                Z[i, j] = g(xdata[i], *a)
        
        for i in range(m):
            for j in range(i, m):
                C[i,j] = 0
                for l in range(n):
                    C[i,j] += Z[l,i]*Z[l,j]
                C[j,i] = C[i,j]

        for i in range(m):
            d[i] = 0
            for l in range(n):
                d[i] += Z[l,i]*D[l]

        da = gauss_elimination(C, d, m)
        an = a + da
        er = linalg.norm(da)/linalg.norm(a)
        k += 1
    return an