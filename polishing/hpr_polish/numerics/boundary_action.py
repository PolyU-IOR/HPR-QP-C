"""Structural numerical primitives; no experiment runner or standalone optimizer."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la
from .poisson import tensor,poisson
from .native import projection,norm


class BoundaryAction:
    def __init__(self,mp,e,I,grid):
        self.mp=mp;self.I=np.asarray(I);self.grid=grid;m=mp.m
        self.l=m.linear_m;assert len(I)==self.l and grid*grid==self.l
        assert np.all(e['box_cell'][I]==0), 'active state bound forbids this bulk pivot'
        assert np.all(m.al==m.au)
        free=np.flatnonzero(e['box_cell']==0);mark=np.zeros(mp.n,bool);mark[I]=True
        self.B=free[~mark[free]];self.qs=m.q[I];self.qb=m.q[self.B]
        self.K=m.A[:self.l,I].tocsr();delta=self.K-tensor(grid);delta.eliminate_zeros();assert not delta.nnz
        self.L=m.A[:self.l,self.B].tocsr();self.C=m.A[self.l:,I].tocsr();self.R=m.A[self.l:,self.B].tocsr()
        _,_,S=projection(m,e['raw_y'],True);self.S=S[self.l:,self.l:].tocsr();self.G=sp.eye(self.S.shape[0],format='csr')-self.S
        self.size=len(self.B)+self.S.shape[0];self.last=None
    def action(self,z):
        tick=time.perf_counter();u=z[:len(self.B)];y=z[len(self.B):]
        f=-self.C.T@y;g=self.L@u
        ds=poisson(g,self.grid);dl=poisson(self.qs*ds-f,self.grid)
        out=np.r_[self.qb*u-self.R.T@y+self.L.T@dl,self.G@(self.R@u-self.C@ds)+self.mp.tau*(self.S@y)]
        self.last=dict(f=f,g=g,ds=ds,dl=dl,seconds=time.perf_counter()-tick);return out
    def audit(self):
        x=self.last;ld=np.longdouble;ds=x['ds'].astype(ld);dl=x['dl'].astype(ld)
        res=np.r_[self.qs.astype(ld)*ds-self.K.T@dl-x['f'],self.K@ds-x['g']]
        return float(np.sqrt(np.sum(res*res)))/max(1,norm(np.r_[x['f'],x['g']]))
