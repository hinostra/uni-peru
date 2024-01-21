#! /usr/bin/python3

from numpy import *

import warnings
warnings.filterwarnings("ignore")

# Ax = b
# A : matriz cuadrada de orden n

#### SUSTITUCIÓN DIRECTA (FORWARD) ####
# A: matriz triangular inferior L, A[n,n] != 0
def forward_substitution(A, b, n):
    x = zeros_like(b, dtype=float)
    bc = copy(b)
    for i in range(n):
        for j in range(i):
            bc[i] -= A[i,j]*x[j]
        x[i] = bc[i]/A[i,i]          # x[k] = (b[k] - sum A[k,j]*x[j])/A[k,k], j=1,...,k-1
    return x

''' MÉTODOS ITERATIVOS '''

#### MÉTODO DE JACOBI ####
# A[i,i] != 0
def jacobi(A, b, n, emax=1e-4, nmax=50):
    k = 0
    err = list()
    x = zeros_like(b, dtype=float)          # x[k]
    xn = zeros_like(b, dtype=float)         # x[k+1]
    er = 2*emax
    while er > emax and k < nmax:
        x = copy(xn)
        for i in range(n):
            xn[i] = b[i]/A[i,i]                 # fórmula de recurrencia
            for j in range(n):
                if j != i:
                    xn[i] -= A[i,j]*x[j]/A[i,i]
        er = linalg.norm(abs(xn - x))/linalg.norm(x)
        err.append(er)
        k += 1
    return xn, err


#### MÉTODO GAUSS-SEIDEL ####
def gauss_seidel(A, b, n, emax=1e-4, nmax=50):
    k = 0
    err = list()
    x = zeros_like(b, dtype=float)
    xn = zeros_like(b, dtype=float)
    bc = zeros_like(b, dtype=float)
    er = 2*emax
    while er > emax and k < nmax:
        x = copy(xn)
        for i in range(n):
            bc[i] = b[i]
            for j in range(i+1, n):
                bc[i] -= A[i,j]*x[j]
        xn = forward_substitution(A, bc, n)
        er = linalg.norm(abs(xn - x))/linalg.norm(x)
        err.append(er)
        k += 1
    return xn, err


#### SUCCESSIVE OVER RELAXATION (SOR) ####
# w: parámetro de relajación
def sor(A, b, w, n, emax=1e-4, nmax=50):
    k = 0
    err = list()
    x = zeros_like(b, dtype=float)
    xn = zeros_like(b, dtype=float)
    bc = zeros_like(b, dtype=float)
    er = 2*emax
    while er > emax and k < nmax:
        x = copy(xn)
        bc = copy(b)
        for i in range(n):
            for j in range(i):
                bc[i] -= (1-w)*A[i,j]*x[j]
            for j in range(i+1,n):
                bc[i] -= A[i,j]*x[j]
        # xn = w*forward_substitution(w*copy(A), bc, n)
        for i in range(n):
            for j in range(i):
                bc[i] -= w*A[i,j]*xn[j]
            xn[i] = bc[i]/A[i,i]
        er = linalg.norm(abs(xn - x))/linalg.norm(x)
        err.append(er)
        k += 1
    return xn, err

''' MÉTODOS DE OPTIMIZACIÓN '''

#### MÉTODO DE MÁXIMO DESCENSO ####
def maximo_descenso(A, b, n, emax=1e-4, nmax=50):
    k = 0
    err = list()
    x = zeros_like(b, dtype=float)
    xn = zeros_like(b, dtype=float)
    r = - copy(b)
    alpha = 1
    er = 2*emax
    while er > emax and k < nmax:
        x = copy(xn)
        # r = dot(A, x) - b
        for i in range(n):
            r[i] = -b[i]
            for j in range(n):
                r[i] += A[i, j]*x[j]
        # alpha = dot(r, r)/dot(r, dot(A, r))
        N, D = 0, 0
        for i in range(n):
            N += r[i]*r[i]
            for j in range(n):
                D += r[i]*A[i,j]*r[j]
        alpha = N/D
        # xn = x - dot(a, r)
        for i in range(n):
            xn[i] = x[i] - alpha*r[i]
        er = linalg.norm(abs(xn - x))/linalg.norm(x)
        err.append(er)
        k += 1
    return xn, err

#### MÉTODO DE GRADIENTE CONJUGADO ####
def gradiente_conjugado(A, b, n, emax=1e-4):
    k = 0
    err = list()
    # x(k), x(k+1)
    x = zeros_like(b, dtype=float)
    xn = zeros_like(b, dtype=float)
    # r(k), r(k+1)
    r = -1*copy(b)
    rn = -1*copy(b)
    # p(k), p(k+1)
    p = copy(b)
    pn = copy(b)
    alpha, beta = 1, 1
    er = 2*emax
    while er > emax and k < n:
        x = copy(xn)
        p = copy(pn)
        r = copy(rn)
        # alpha = dot(r,r)/dot(p, dot(A,p))
        Na, Da, alpha = 0., 0., 0.
        for i in range(n):
            Na += r[i]*r[i]
            for j in range(n):
                Da += p[i]*A[i,j]*p[j]
        alpha = Na/Da
        # xn = x + a*p
        for i in range(n):
            xn[i] = x[i] + alpha*p[i]
        # rn = r + a*dot(A, p)
        for i in range(n):
            rn[i] = r[i]
            for j in range(n):
                rn[i] += alpha*A[i,j]*p[j]
        # beta = dot(rn, rn)/dot(r,r)
        Nb, Db, beta = 0., 0., 0.
        for i in range(n):
            Nb += rn[i]*rn[i]
            Db += r[i]*r[i]
        beta = Nb/Db
        for i in range(n):
            pn[i] = -rn[i] + beta*p[i]
        er = linalg.norm(abs(xn - x))/linalg.norm(x)
        err.append(er)
        k += 1
    return xn, err