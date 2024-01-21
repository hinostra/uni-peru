#! /usr/bin/python3
from re import A
from numpy import *

import warnings
warnings.filterwarnings("ignore")

def potencias(A, x0, accuracy=1e-4, nmax=10):
    # n = len(x0)
    wold = zeros_like(x0)
    w = zeros_like(x0)
    
    w = dot(A, x0)
    w /= max(w)
    eigen = dot(dot(A,w), w)/dot(w,w)
    es, k = 2*accuracy, 0
    while es > accuracy and k<=nmax:
        wold = copy(w)
        eigenold = eigen
        w = dot(A, wold)
        w /= max(w)
        eigen = dot(dot(A,w), w)/dot(w,w)
        es = abs(eigen-eigenold)/abs(eigen)
        k += 1
    return eigen

# A = array([[5, -2], [-2, 8]], float)
# x0 = array([1., 1.])

# autovalor = potencias(A, x0, 1e-6, 15)
# print(autovalor)