#! /usr/bin/python3
from numpy import *

import warnings
warnings.filterwarnings("ignore")

''' SISTEMAS DE EDOS '''
# Filas: x, y1, y2, ..., ym
# Columnas: x[0], x[1], ..., x[n]

''' MÉTODO DE EULER '''
# f(x,y) = dy/dx
# dx: ancho de paso
# [xi, xf]: punto inicial y final
# y0: valor inicial
def euler(f, xlims, dx, y0):
    x0, xn = xlims
    n = int(abs(xn-x0)/dx)
    m = len(y0)
    sol = zeros([m+1, n+1], float)
    sol[0,0], sol[1:,0] = x0, y0
    for i in range(n):
        sol[0,i+1] = sol[0,i] + dx
        sol[1:,i+1] = sol[1:,i] + dx*f(*sol[:,i])
    return sol

''' MÉTODO DE HEUN '''
# f(x,y) = dy/dx
# dx: ancho de paso
# [xi, xf]: punto inicial y final
# y0: valor inicial
def heun_simple(f, xlims, dx, y0):
    x0, xn = xlims
    n = int(abs(xn-x0)/dx)
    m = len(y0)
    sol = zeros([m+1, n+1], float)
    sol[0,0], sol[1:,0] = x0, y0
    for i in range(n):
        sol[0,i+1] = sol[0,i] + dx
        ym = sol[1:,i] + dx*f(*sol[:,i])
        fm = (f(*sol[:,i]) + f(sol[0,i+1], *ym))/2
        sol[1:,i+1] = sol[1:,i] + dx*fm
    return sol

# Método de Heun iterativo
def heun_iteration(f, xlims, dx, y0, emax=1e-4, nmax=20):
    x0, xn = xlims
    n = int(abs(xn-x0)/dx)
    m = len(y0)
    sol = zeros([m+1, n+1], float)
    sol[0,0], sol[1:,0] = x0, y0
    for i in range(n):
        sol[0,i+1] = sol[0,i] + dx
        ynew = sol[1:,i] + dx*f(*sol[:,i])
        es, k = 2*emax, 0
        while es > emax and k <= nmax:
            yold = copy(ynew)
            fm = (f(*sol[:,i]) + f(sol[0,i+1], *yold))/2
            ynew = sol[1:,i] + dx*fm
            es = max(abs(ynew - yold)/abs(yold))*100
            k += 1
        sol[1:,i+1] = copy(ynew)
    return sol

''' MÉTODO DEL PUNTO MEDIO '''
# f(x,y) = dy/dx
# dx: ancho de paso
# [xi, xf]: punto inicial y final
# y0: valor inicial
def midpoint(f, xlims, dx, y0):
    x0, xn = xlims
    n = int(abs(xn-x0)/dx)
    m = len(y0)
    sol = zeros([m+1, n+1], float)
    sol[0,0], sol[1:,0] = x0, y0
    for i in range(n):
        xhalf = sol[0,i] + dx/2
        sol[0,i+1] = sol[0,i] + dx
        yhalf = sol[1:,i] + f(*sol[:,i])*dx/2
        sol[1:,i+1] = sol[1:,i] + f(xhalf, *yhalf)*dx
    return sol

''' MÉTODOS DE RUNGE-KUTTA '''
# f(x,y) = dy/dx
# dx: ancho de paso
# [xi, xf]: punto inicial y final
# y0: valor inicial

def RK2(f, xlims, dx, y0, tipo='Heun'):
    x0, xn = xlims
    n = int(abs(xn-x0)/dx)
    m = len(y0)
    sol = zeros([m+1, n+1], float)
    sol[0,0], sol[1:,0] = x0, y0
    if tipo == 'Heun':
        a1, a2 = 1/2, 1/2
    elif tipo == 'Midpoint':
        a1, a2 = 0, 1
    elif tipo == 'Ralston':
        a1, a2 = 1/3, 2/3
    p1, q11 = 1/(2*a2), 1/(2*a2)
    for i in range(n):
        sol[0,i+1] = sol[0,i] + dx
        k1 = f(*sol[:,i])
        ym = sol[1:,i] + q11*k1*dx
        k2 = f(sol[0,i] + p1*dx, *ym)
        sol[1:,i+1] = sol[1:,i] + (a1*k1 + a2*k2)*dx
    return sol

def RK3(f, xlims, dx, y0):
    x0, xn = xlims
    n = int(abs(xn-x0)/dx)
    m = len(y0)
    sol = zeros([m+1, n+1], float)
    sol[0,0], sol[1:,0] = x0, y0
    for i in range(n):
        sol[0,i+1] = sol[0,i] + dx
        k1 = f(*sol[:,i])
        ym = sol[1:,i] + 1/2*k1*dx
        k2 = f(sol[0,i] + dx/2, *ym)
        ym = sol[1:,i] - k1*dx + 2*k2*dx
        k3 = f(sol[0,i] + dx, *ym)
        sol[1:,i+1] = sol[1:,i] + 1/6*(k1 + 4*k2 + k3)*dx
    return sol

def RK4(f, xlims, dx, y0):
    x0, xn = xlims
    n = int(abs(xn-x0)/dx)
    m = len(y0)
    sol = zeros([m+1, n+1], float)
    sol[0,0], sol[1:,0] = x0, y0
    for i in range(n):
        sol[0,i+1] = sol[0,i] + dx
        k1 = f(*sol[:,i])
        ym = sol[1:,i] + 1/2*k1*dx
        k2 = f(sol[0,i] + dx/2, *ym)
        ym = sol[1:,i] + 1/2*k2*dx
        k3 = f(sol[0,i] + dx/2, *ym)
        ym = sol[1:,i] + k3*dx
        k4 = f(sol[0,i] + dx, *ym)
        sol[1:,i+1] = sol[1:,i] + 1/6*(k1 + 2*k2 + 2*k3 + k4)*dx
    return sol

def RK5(f, xlims, dx, y0):
    x0, xn = xlims
    n = int(abs(xn-x0)/dx)
    m = len(y0)
    sol = zeros([m+1, n+1], float)
    sol[0,0], sol[1:,0] = x0, y0
    for i in range(n):
        sol[0,i+1] = sol[0,i] + dx
        k1 = f(*sol[:,i])
        ym = sol[1:,i] + 1/4*k1*dx
        k2 = f(sol[0,i] + dx/4, *ym)
        ym = sol[1:,i] + 1/8*k1*dx + 1/8*k2*dx
        k3 = f(sol[0,i] + dx/4, *ym)
        ym = sol[1:,i] - 1/2*k2*dx + k3*dx
        k4 = f(sol[0,i] + dx/2, *ym)
        ym = sol[1:,i] + 3/16*k1*dx + 9/16*k4*dx
        k5 = f(sol[0,i] + 3*dx/4, *ym)
        ym = sol[1:,i] - 3/7*k1*dx + 2/7*k2*dx + 12/7*k3*dx -12/7*k4*dx + 8/7*k5*dx
        k6 = f(sol[0,i] + dx, *ym)
        sol[1:,i+1] = sol[1:,i] + 1/90*(7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)*dx
    return sol

''' MÉTODO DE HEUN MODIFICADO '''
def heun_modificado(f, xlims, dx, y0, emax=1e-4, nmax=20):
    x0, xn = xlims
    n = int(abs(xn-x0)/dx)
    m = len(y0)
    sol = zeros([m+1, n+1], float)
    sol[:,0:2] = heun_simple(f, [x0,x0+dx], dx, y0)
    for i in range(1, n):
        sol[0,i+1] = sol[0,i] + dx
        ynew = sol[1:,i-1] + f(*sol[:,i])*2*dx
        es, k = 2*emax, 0
        while es > emax and k <= nmax:
            yold = copy(ynew)
            fm = (f(*sol[:,i]) + f(sol[0,i+1], *yold))/2
            ynew = sol[1:,i] + fm*dx
            es = max(abs(ynew - yold)/abs(yold))*100
            k += 1
        sol[1:,i+1] = copy(ynew)
    return sol


''' FÓRMULAS DE ADAM-BASHFORTH '''

# Coeficientes del predictor
def predictor_ab(k):
    if k==1:
        return [1.]
    elif k==2:
        return [3/2, -1/2]
    elif k==3:
        return [23/12, -16/12, 5/12]
    elif k==4:
        return [55/24, -59/24, 37/24, -9/24]
    elif k==5:
        return [1901/720, -2774/720, 2616/720, -1274/720, 251/720]
    elif k==6:
        return [4277/720, -7923/720, 9982/720, -7298/720, 2877/720, -475/720]

def corrector_ab(k):
    if k==2:
        return [1/2, 1/2]
    elif k==3:
        return [5/12, 8/12, -1/12]
    elif k==4:
        return [9/24, 19/24, -5/24, 1/24]
    elif k==5:
        return [251/720, 646/720, -264/720, 106/720, -19/720]
    elif k==6:
        return [475/1440, 1427/1440, -798/1440, 482/1440, -173/1440, 27/1440]

def adam_bashforth(f, xlims, dx, y0, k=1, emax=1e-4, nmax=20):
    x0, xn = xlims
    n = int(abs(xn-x0)/dx)
    m = len(y0)
    beta = predictor_ab(k)
    alpha = corrector_ab(k)
    sol = zeros([m+1, n+1], float)
    sol[:,0:k+1] = heun_simple(f, [x0,x0+k*dx], dx, y0)
    for i in range(k, n):
        sol[0,i+1] = sol[0,i] + dx
        fm = zeros(m)
        for j in range(k):
            fm += beta[j]*f(*sol[:,i-k])
        ynew = sol[1:,i] + dx*fm
        es, n = 2*emax, 0
        while es > emax and n <= nmax and k>1:
            yold = copy(ynew)
            fm = alpha[0]*copy(f(sol[0,i+1],*yold))
            for j in range(1,k):
                fm += alpha[j]*f(*sol[:,i+1-k])
            ynew = sol[1:,i] + dx*fm
            es = max(abs(ynew - yold)/abs(yold))*100
            n += 1
        sol[1:,i+1] = copy(ynew)
    return sol