"""Structural numerical primitives; no experiment runner or standalone optimizer."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la
from dataclasses import dataclass


def norm(v):return float(np.linalg.norm(v))

def maxabs(v):return float(np.max(np.abs(v),initial=0))

@dataclass
class Model:
    q:np.ndarray
    A:sp.csr_matrix
    c:np.ndarray
    lo:np.ndarray
    hi:np.ndarray
    al:np.ndarray
    au:np.ndarray
    rhs:np.ndarray
    linear_m:int
    offsets:np.ndarray
    constant:float=0.
    name:str='model'

def cone_project(v):
    """Disjoint rank1 bridge cones: v has shape(k,3)."""
    t=v[:,0];r=np.linalg.norm(v[:,1:],axis=1)
    inside=r<=t;outside=(r<=-t)&~inside;middle=~(inside|outside)
    p=np.zeros_like(v);p[inside]=v[inside]
    alpha=.5*(1+t[middle]/r[middle]);p[middle,0]=.5*(r[middle]+t[middle]);p[middle,1:]=alpha[:,None]*v[middle,1:]
    return p,np.where(inside,1,np.where(outside,-1,0))

def projection(m,v,derivative=False):
    p=np.empty_like(v);l=m.linear_m
    p[:l]=np.clip(v[:l],m.al,m.au)
    cones=v[l:].reshape(-1,3);pc,kind=cone_project(cones);p[l:]=pc.ravel()
    if not derivative:return p,kind
    linear=((v[:l]>m.al)&(v[:l]<m.au)).astype(float)
    blocks=np.zeros((len(cones),3,3));blocks[kind==1]=np.eye(3)
    idx=np.flatnonzero(kind==0)
    if len(idx):
        t=cones[idx,0];z=cones[idx,1:];r=np.linalg.norm(z,axis=1);unit=z/r[:,None]
        blocks[idx,0,0]=.5;blocks[idx,0,1:]=.5*unit;blocks[idx,1:,0]=.5*unit
        blocks[idx,1:,1:]=.5*(1+t/r)[:,None,None]*np.eye(2)-.5*(t/r)[:,None,None]*unit[:,:,None]*unit[:,None,:]
    rr=(l+3*np.arange(len(cones)))[:,None,None]+np.arange(3)[None,:,None]+np.zeros((1,1,3),int)
    cc=(l+3*np.arange(len(cones)))[:,None,None]+np.arange(3)[None,None,:]+np.zeros((1,3,1),int)
    S=sp.csr_matrix((np.r_[linear,blocks.ravel()],(np.r_[np.arange(l),rr.ravel()],np.r_[np.arange(l),cc.ravel()])),shape=(len(v),len(v)))
    S.eliminate_zeros()
    return p,kind,S

class Native:
    def __init__(self,m,sigma=1.):
        self.m=m;self.n=len(m.q);self.rows=m.A.shape[0];self.size=2*self.n+self.rows
        self.sigma=sigma;self.d=1/(1+sigma*m.q)
        self.lam=1.1*float(np.max(np.asarray(abs(m.A).sum(axis=0))))*float(np.max(np.asarray(abs(m.A).sum(axis=1))))
        assert self.lam>0;self.tau=sigma*self.lam;self.calls=0
        self.bnorm=max(maxabs(m.al[np.isfinite(m.al)]),maxabs(m.au[np.isfinite(m.au)]),maxabs(m.rhs))
    def evaluate(self,v):
        self.calls+=1;m=self.m;n=self.n;sig=self.sigma;x,w,y=v[:n],v[n:2*n],v[2*n:]
        raw=x+sig*(m.A.T@y-m.q*w-m.c);p=np.clip(raw,m.lo,m.hi);z=(p-raw)/sig;xh=2*p-x
        w1=self.d*(sig*m.q*w+xh)
        temp=xh+sig*(m.q*w-m.q*w1) # literal source, not simplified w1
        rv=m.A@temp-m.rhs-self.tau*y;s,kind=projection(m,rv);yb=(s-rv)/self.tau
        wb=w1+sig*self.d*(m.A.T@(yb-y))
        T=np.r_[xh,2*wb-w,2*yb-y];F=v-T
        return dict(F=F,T=T,norm=norm(F),p=p,z=z,y=yb,w=wb,s=s,raw=raw,raw_y=rv,
            box_cell=np.where(raw<=m.lo,-1,np.where(raw>=m.hi,1,0)),cone_cell=kind,
            temp_identity=maxabs(temp-w1))
    def certificate(self,e):
        m=self.m;x=e['p'];y=e['y'];z=e['z'];Qx=m.q*x;Ax=m.A@x;Aty=m.A.T@y
        p=float(m.c@x+.5*x@Qx+m.constant)
        d=float(-.5*x@Qx+z@x+y@(e['s']+m.rhs)+m.constant)
        gap=abs(p-d)/(1+max(abs(p),abs(d)))
        rp=maxabs(projection(m,Ax-m.rhs)[0]-(Ax-m.rhs))/(1+max(self.bnorm,maxabs(Ax)))
        rp=max(rp,maxabs(np.clip(x,m.lo,m.hi)-x))
        rd=maxabs(Qx+m.c-Aty-z)/(1+max(maxabs(m.c),maxabs(Qx),maxabs(Aty)))
        return dict(error=max(rp,rd,gap),rp=rp,rd=rd,gap=gap,primal=p,dual=d)
    def jacobian(self,e):
        m=self.m;n=self.n;r=self.rows;sig=self.sigma
        In=sp.eye(n,format='csr');Ir=sp.eye(r,format='csr');zn=sp.csr_matrix((n,n));zr=sp.csr_matrix((n,r))
        X=sp.hstack([In,zn,zr],format='csr');W=sp.hstack([zn,In,zr],format='csr');Y=sp.hstack([sp.csr_matrix((r,2*n)),Ir],format='csr')
        E=sp.diags((e['box_cell']==0).astype(float));D=sp.diags(self.d);Q=sp.diags(m.q)
        P=E@(X-sig*Q@W+sig*m.A.T@Y);XH=2*P-X;W1=D@(sig*Q@W+XH)
        TMP=XH+sig*(Q@W-Q@W1)
        _,_,S=projection(m,e['raw_y'],True)
        YB=(S-Ir)@(m.A@TMP-self.tau*Y)/self.tau
        WB=W1+sig*D@m.A.T@(YB-Y)
        J=sp.eye(self.size,format='csr')-sp.vstack([XH,2*WB-W,2*YB-Y],format='csr')
        J.eliminate_zeros()
        return J
