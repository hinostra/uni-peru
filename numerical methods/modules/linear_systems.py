#! /usr/bin/python3

from numpy import *

import warnings
warnings.filterwarnings("ignore")

# Ax = b
# A : matriz cuadrada de orden N

''' MÉTODOS DIRECTOS '''

#### SUSTITUCIÓN DIRECTA (FORWARD) ####
# A: matriz triangular inferior L, A[N,N] != 0
def forward_substitution(A, b, N):
    x = zeros_like(b, dtype=float)
    bc = copy(b)
    for i in range(N):
        for j in range(i):
            bc[i] -= A[i,j]*x[j]
        x[i] = bc[i]/A[i,i]          # x[k] = (b[k] - sum A[k,j]*x[j])/A[k,k], j=1,...,k-1
    return x


#### SUSTITUCIÓN DIRECTA (BACKWARD) ####
# A: matriz triangular superior U, A[N,N] != 0
def backward_substitution(A, b, N):
    x = zeros_like(b, dtype=float)
    bc = copy(b)
    for i in range(N-1,-1,-1):
        for j in range(i+1,N):
            bc[i] -= A[i,j]*x[j]
        x[i] = bc[i]/A[i,i]          # x[k] = (b[k] - A[k,j]*x[j])/A[k,k], j=k+1,...,N
    return x


#### ELIMINACIÓN GAUSSIANA CON PIVOT ####
# A: matriz triangular superior U, A[N,N] != 0

# Pivoteo parcial en el elemento a[k, k]
def pivot(A, b, k, down=True):
    p = k
    big = abs(A[k, k])
    if down:
        for ii in range(k+1, len(b)):
            dummy = abs(A[ii, k])
            if dummy > big:
                big = dummy
                p = ii
        if p != k:
            for jj in range(k, len(b)):
                dummy = A[k, jj]
                A[k, jj] = A[p, jj]
                A[p, jj] = dummy
            dummy = b[p]
            b[p] = b[k]
            b[k] = dummy
    else:
        for ii in range(k, -1, -1):
            dummy = abs(A[ii, k])
            if dummy > big:
                big = dummy
                p = ii
        if p != k:
            for jj in range(len(b)):
                dummy = A[k, jj]
                A[k, jj] = A[p, jj]
                A[p, jj] = dummy
            dummy = b[p]
            b[p] = b[k]
            b[k] = dummy

# Eliminación gausiana
def gauss_elimination(A, b, N, e=1e-6):
    x = zeros_like(b, dtype=float)
    Aa = copy(A)
    ba = copy(b)
    for i in range(N):
        if abs(Aa[i, i]) < e:
            pivot(Aa, ba, i)
        for j in range(i+1, N):
            factor = Aa[j,i]/Aa[i,i]
            for k in range(len(b)):
                Aa[j,k] -= factor*Aa[i,k]
            ba[j] -= factor*ba[i]
    x = backward_substitution(Aa, ba, N)
    return x


#### DESCOMPOSICIÓN LU ####
# A: matriz regular
# A = LU

def LU_decomposition(A, N):
    L = eye(N, dtype=float)
    U = zeros_like(A, dtype=float)
    for j in range(N):
        for i in range(j+1):
            U[i, j] = A[i, j]
            for k in range(i):
                U[i, j] -= L[i, k]*U[k, j]
        for i in range(j+1, N):
            L[i, j] = A[i, j]/U[j, j]
            for k in range(j):
                L[i, j] -= L[i, k]*U[k, j]/U[j, j]
    return L, U

def LU_solution(L, U, b, N):
    d = forward_substitution(L, b, N)
    x = backward_substitution(U, d, N)
    return x


#### DESCOMPOSICIÓN DE CHOLESKY ####
# A: Matriz simétrica y definida positiva
# A = G^T G

def cholesky_decomposition(A, N):
    G = zeros_like(A, dtype=float)
    for i in range(N):
        S = 0
        for k in range(i):
            S += G[k, i]**2
        G[i, i] = sqrt(A[i, i] - S)

        for j in range(i+1, N):
            G[i, j] = A[i, j]/G[i, i]
            for k in range(i):
                G[i, j] -= G[k, i]*G[k, j]/G[i, i]
    return G

def cholesky_solution(A, b, N):
    G = cholesky_decomposition(A, N)
    x = LU_solution(transpose(G), G, b, N)
    return x


''' MÉTODOS ITERATIVOS '''

#### MÉTODO DE JACOBI ####
# A[i,i] != 0
def jacobi(A, b, N, emax=1e-4, nmax=50):
    k = 0
    err = list()
    x = zeros_like(b, dtype=float)          # x[k]
    xn = zeros_like(b, dtype=float)         # x[k+1]
    er = 2*emax
    while er > emax and k < nmax:
        x = copy(xn)
        for i in range(N):
            xn[i] = b[i]/A[i,i]                 # fórmula de recurrencia
            for j in range(N):
                if j != i:
                    xn[i] -= A[i,j]*x[j]/A[i,i]
        er = linalg.norm(abs(xn - x))/linalg.norm(x)
        err.append(er)
        k += 1
    return xn, err


#### MÉTODO GAUSS-SEIDEL ####
def gauss_seidel(A, b, N, emax=1e-4, nmax=50):
    k = 0
    err = list()
    x = zeros_like(b, dtype=float)
    xn = zeros_like(b, dtype=float)
    bc = zeros_like(b, dtype=float)
    er = 2*emax
    while er > emax and k < nmax:
        x = copy(xn)
        for i in range(N):
            bc[i] = b[i]
            for j in range(i+1, N):
                bc[i] -= A[i,j]*x[j]
        xn = forward_substitution(A, bc, N)
        er = linalg.norm(abs(xn - x))/linalg.norm(x)
        err.append(er)
        k += 1
    return xn, err


#### SUCCESSIVE OVER RELAXATION (SOR) ####
# w: parámetro de relajación
def sor(A, b, w, N, emax=1e-4, nmax=50):
    k = 0
    err = list()
    x = zeros_like(b, dtype=float)
    xn = zeros_like(b, dtype=float)
    bc = zeros_like(b, dtype=float)
    er = 2*emax
    while er > emax and k < nmax:
        x = copy(xn)
        bc = copy(b)
        for i in range(N):
            for j in range(i):
                bc[i] -= (1-w)*A[i,j]*x[j]
            for j in range(i+1,N):
                bc[i] -= A[i,j]*x[j]
        # xn = w*forward_substitution(w*copy(A), bc, N)
        for i in range(N):
            for j in range(i):
                bc[i] -= w*A[i,j]*xn[j]
            xn[i] = bc[i]/A[i,i]
        er = linalg.norm(abs(xn - x))/linalg.norm(x)
        err.append(er)
        k += 1
    return xn, err

''' MÉTODOS DE OPTIMIZACIÓN '''
# A simétrica definida positiva

#### MÉTODO DE MÁXIMO DESCENSO ####
def maximo_descenso(A, b, N, emax=1e-4, nmax=50):
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
        for i in range(N):
            r[i] = -b[i]
            for j in range(N):
                r[i] += A[i, j]*x[j]
        # alpha = dot(r, r)/dot(r, dot(A, r))
        N, D = 0, 0
        for i in range(N):
            N += r[i]*r[i]
            for j in range(N):
                D += r[i]*A[i,j]*r[j]
        alpha = N/D
        # xn = x - dot(a, r)
        for i in range(N):
            xn[i] = x[i] - alpha*r[i]
        er = linalg.norm(abs(xn - x))/linalg.norm(x)
        err.append(er)
        k += 1
    return xn, err

#### MÉTODO DE GRADIENTE CONJUGADO ####
def gradiente_conjugado(A, b, N, emax=1e-4):
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
    while er > emax and k < N:
        x = copy(xn)
        p = copy(pn)
        r = copy(rn)
        # alpha = dot(r,r)/dot(p, dot(A,p))
        Na, Da, alpha = 0., 0., 0.
        for i in range(N):
            Na += r[i]*r[i]
            for j in range(N):
                Da += p[i]*A[i,j]*p[j]
        alpha = Na/Da
        # xn = x + a*p
        for i in range(N):
            xn[i] = x[i] + alpha*p[i]
        # rn = r + a*dot(A, p)
        for i in range(N):
            rn[i] = r[i]
            for j in range(N):
                rn[i] += alpha*A[i,j]*p[j]
        # beta = dot(rn, rn)/dot(r,r)
        Nb, Db, beta = 0., 0., 0.
        for i in range(N):
            Nb += rn[i]*rn[i]
            Db += r[i]*r[i]
        beta = Nb/Db
        for i in range(N):
            pn[i] = -rn[i] + beta*p[i]
        er = linalg.norm(abs(xn - x))/linalg.norm(x)
        err.append(er)
        k += 1
    return xn, err