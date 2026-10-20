"""Structural numerical primitives; no experiment runner or standalone optimizer."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la
from scipy.fft import dstn,idstn


def tensor(n):
    T=sp.diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1],format='csr')
    return sp.kronsum(T,T,format='csr')

def poisson(rhs,n):
    lam=4*np.sin(np.pi*np.arange(1,n+1)/(2*(n+1)))**2
    a=dstn(np.asarray(rhs,float).reshape(n,n),type=1,norm='ortho',workers=1)
    return idstn(a/(lam[:,None]+lam[None,:]),type=1,norm='ortho',workers=1).ravel()
