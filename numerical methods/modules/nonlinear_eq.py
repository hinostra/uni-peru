#! /usr/bin/python3

import warnings
warnings.filterwarnings("ignore")

''' MÉTODOS ITERATIVOS PARA SOLUCIONAR ECUACIONES NO LINEALES '''

#### BÚSQUEDA INCREMENTAL ####
# f: Función
# xl: Punto extremo inferior inicial
# h: Paso
# nmax: Número máximo de iteraciones

def incremental_search(f, xl, h=1e-2, nmax=50):
    k = 0               # Contador de iteraciones
    xu = xl + h         # Extremo superior inicial
    while f(xl)*f(xu) > 0 and k < nmax:
        xl = xu         # Nuevo extremo inferior
        xu = xl + h     # Nuevo extremo superior
        k += 1
    return [xl, xu]


#### BISECCIÓN ####
# f: Función
# xl: Punto extremo inferior inicial
# xu: Punto extremo superior inicial
# es: Error relativo mínimo (precisión %)
# nmax: Número máximo de iteraciones
# error: Retornar lista de errores?

def bisection(f, xl, xu, es=1e-4, nmax=50):
    k = 1                                   # Contador de iteraciones
    xr = (xl + xu)/2                        # xr inicial
    err = list()                            # Lista para los errores
    if xr != 0:
        ea = abs((xr - xl)/xr)*100          # Error relativo inicial
    else:
        ea = abs((xr - xl)/xl)*100
    err.append(ea)
    while ea > es and k < nmax:
        if f(xl)*f(xr) < 0:
            xu = xr                         # Nuevo extremo superior
        if f(xu)*f(xr) < 0:
            xl = xr                         # Nuevo extremo inferior
        if f(xr) == 0:
            return xr, err
        xrold = xr                          # Punto medio inicial
        xr = (xl + xu)/2                    # Punto medio final
        if xr != 0:
            ea = abs((xr - xrold)/xr)*100   # Error relativo %
        else:
            ea = abs((xr - xrold)/xrold)*100
        err.append(ea)
        k += 1
    return xr, err


#### FALSA POSICIÓN ####
# f: Función
# xl: Punto extremo inferior inicial
# xu: Punto extremo superior inicial
# es: Error relativo mínimo (precisión %)
# nmax: Número máximo de iteraciones
# error: Retornar lista de errores?

def false_position(f, xl, xu, es=1e-4, nmax=50):
    k = 1                                       # Contador de iteraciones
    xr = xu - f(xu)*(xl-xu)/(f(xl)-f(xu))       # xr inicial
    err = list()                                # Lista para los errores
    if xr != 0:
        ea = abs((xr - xl)/xr)*100              # Error relativo % inicial
    else:
        ea = abs((xr - xl)/xl)*100
    err.append(ea)
    while ea > es and k < nmax:
        if f(xl)*f(xr) < 0:
            xu = xr                             # Nuevo extremo superior
        if f(xu)*f(xr) < 0:
            xl = xr                             # Nuevo extremo inferior
        if f(xr) == 0:
            return xr, err
        xrold = xr                              # Punto xr inicial
        xr = xu - f(xu)*(xl-xu)/(f(xl)-f(xu))   # Punto xr final
        if xr != 0:
            ea = abs((xr - xrold)/xr)*100       # Error relativo %
        else:
            ea = abs((xr - xrold)/xrold)*100
        err.append(ea)
        k += 1
    return xr, err


#### PUNTO FIJO ####
# g: Función x = g(x)
# x0: Punto estimado inicial
# es: Error relativo mínimo (precisión %)
# nmax: Número máximo de iteraciones
# error: Retornar lista de errores?

def fixed_point(g, x0, es=1e-4, nmax=50):
    k = 1                               # Contador de iteraciones
    x = x0
    xn = g(x)
    err = list()                        # Lista para los errores
    if xn != 0:
        ea = abs((xn - x)/xn)*100       # Error relativo % inicial
    else:
        ea = abs((xn - x)/x)*100
    err.append(ea)
    while ea > es and k < nmax:
        x = xn                          # x[i] = g(x[i-1])
        xn = g(x)                       # x[i+1] = g(x[i])
        if xn != 0:
            ea = abs((xn - x)/xn)*100   # Error relativo %
        else:
            ea = abs((xn - x)/x)*100
        err.append(ea)
        k += 1
    return xn, err

#### NEWTON-RAPHSON ####
# f: Función
# df: Derivada de f
# x0: Estimación inicial de la raíz
# es: Error relativo mínimo (precisión %)
# nmax: Número máximo de iteraciones
# error: Retornar lista de errores?

def newton_raphson(f, df, x0, es=1e-4, nmax=50):
    k = 1                               # Contador de iteraciones
    x = x0
    xn = x - f(x)/df(x)                 # x1 = x0 - f(x0)/f'(x0)
    err = list()                        # Lista para los errores
    if xn != 0:
        ea = abs((xn - x)/xn)*100       # Error relativo % inicial
    else:
        ea = abs((xn - x)/x)*100
    err.append(ea)
    while ea > es and k < nmax:
        x = xn                          # x[i]: punto xn de la iteración anterior
        xn = x - f(x)/df(x)             # x[i+1]: nuevo punto
        if xn != 0:
            ea = abs((xn - x)/xn)*100   # Error relativo %
        else:
            ea = abs((xn - x)/x)*100
        err.append(ea)
        k += 1
    return xn, err


#### SECANTE ####
# f: Función
# x0: Estimación inicial 1 de la raíz (backward)
# x1: Estimación inicial 2 de la raíz (principal)
# es: Error relativo mínimo (precisión %)
# nmax: Número máximo de iteraciones
# error: Retornar lista de errores?

def secant(f, x0, x1, es=1e-4, nmax=50):
    k = 1                                       # Contador de iteraciones
    x, xb = x1, x0                              # Primeros puntos para derivada
    xn = x - (f(x)*(xb - x))/(f(xb) - f(x))
    err = list()                                # Lista para los errores
    if xn != 0:
        ea = abs((xn - x)/xn)*100               # Error relativo % inicial
    else:
        ea = abs((xn - x)/x)*100
    err.append(ea)
    while ea > es and k < nmax:
        xb = x                                  # x[i-1]: derivada backward
        x = xn                                  # x[i]: punto xn de la iteración anterior
        xn = x - (f(x)*(xb - x))/(f(xb) - f(x)) # x[i+1]: nuevo punto
        if xn != 0:
            ea = abs((xn - x)/xn)*100           # Error relativo %
        else:
            ea = abs((xn - x)/x)*100
        err.append(ea)
        k += 1
    return xn, err