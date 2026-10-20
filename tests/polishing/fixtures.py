"""Analytic QPs and the small spatial model used for structural qualification."""
import numpy as np
import scipy.sparse as sp
from hpr_polish.core import Model

def qp(kind):
    q=np.array([[2.,.3],[.3,1.]])
    optimum=np.array([.25,.5]);c=-q@optimum
    if kind=='affine':a=np.array([[1.,2.]]);lo=hi=a@optimum
    elif kind=='mixed':a=np.array([[0.,1.],[1.,2.],[1.,0.]]);lo=np.array([0.,1.25,-1.]);hi=np.array([1.,1.25,1.])
    elif kind=='dependent':a=np.array([[1.,2.],[2.,4.]]);lo=hi=a@optimum
    elif kind=='no_equalities':a=np.eye(2);lo=np.zeros(2);hi=np.ones(2)
    elif kind=='box':a=np.empty((0,2));lo=hi=np.empty(0)
    else:raise ValueError(kind)
    return Model(q,sp.csr_matrix(a),c,[-1.,-1.],[1.,1.],lo,hi,np.zeros(len(a))),float(.5*optimum@q@optimum+c@optimum)

def spatial(g):
    h=1/(g+1);e=g*g;k=4*g
    t=sp.diags([-np.ones(g-1),2*np.ones(g),-np.ones(g-1)],[-1,0,1],format='csr');K=sp.kronsum(t,t,format='csr')
    br=np.r_[np.arange(g),np.arange(e-g,e),np.arange(0,e,g),np.arange(g-1,e,g)]
    B=sp.csr_matrix((np.ones(k),(br,np.arange(k))),shape=(e,k))
    E=sp.hstack([-B,K,sp.csr_matrix((e,k))],format='csr')
    C=sp.hstack([sp.eye(k),-B.T,-sp.eye(k)],format='csr')
    R=sp.hstack([2*sp.eye(k),sp.csr_matrix((k,e+k))],format='csr')
    order=np.arange(3*k).reshape(3,k).T.ravel();A=sp.vstack([E,sp.vstack([-C,-C,R],format='csr')[order]],format='csr')
    v=np.arange(1,g+1)*h;d=(3-2*((v[:,None]-.5)**2+(v[None,:]-.5)**2)).ravel()
    q=np.r_[np.zeros(k),np.full(e,h*h),np.full(k,.01*h)];c=np.r_[np.zeros(k),-h*h*d,np.zeros(k)]
    return Model(sp.diags(q),A,c,np.r_[np.zeros(k+e),np.full(k,3.7)],np.r_[np.full(k+e,2.071),np.full(k,4.5)],np.r_[np.zeros(e),np.full(3*k,-np.inf)],np.r_[np.zeros(e),np.full(3*k,np.inf)],np.r_[np.zeros(e),np.tile([-1.,1.,0.],k)],tuple((e+3*i,e+3*i+3) for i in range(k)),float(.5*h*h*(d@d)))
