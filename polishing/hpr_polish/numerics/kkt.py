"""Structural numerical primitives; no experiment runner or standalone optimizer."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la
from .native import projection,norm
from .native_audit import replay as native_replay


class KKT:
    def __init__(self,ref):
        self.ref=ref;self.m=ref.m;self.n=ref.n;self.rows=ref.rows;self.size=self.n+self.rows;self.sigma=ref.sigma;self.tau=ref.tau
    def lift(self,u):return np.r_[u[:self.n],u[:self.n],u[self.n:]]
    def evaluate(self,u):
        m=self.m;x=u[:self.n];y=u[self.n:];raw=x-self.sigma*(m.q*x+m.c-m.A.T@y);p=np.clip(raw,m.lo,m.hi)
        ax=m.A@x-m.rhs;rawy=ax-self.tau*y;s,kind=projection(m,rawy);F=np.r_[x-p,ax-s]
        return dict(F=F,norm=norm(F),raw=raw,raw_y=rawy,p=p,s=s,u=u,box_cell=np.where(raw<=m.lo,-1,np.where(raw>=m.hi,1,0)),cone_cell=kind)
    def jacobian(self,e):
        m=self.m;E=sp.diags((e['box_cell']==0).astype(float));_,_,S=projection(m,e['raw_y'],True)
        return sp.bmat([[sp.eye(self.n)-E+self.sigma*E@sp.diags(m.q),-self.sigma*E@m.A.T],[(sp.eye(self.rows)-S)@m.A,self.tau*S]],format='csc')
    def certificate(self,e):return self.ref.certificate(self.ref.evaluate(self.lift(e['u'])))

def replay(mp,u,direction=None):
    m=mp.m;n=mp.n;l=m.linear_m;ld=np.longdouble;x=np.asarray(u[:n],dtype=ld);y=np.asarray(u[n:],dtype=ld);sig=ld(mp.sigma);tau=ld(mp.tau)
    raw=x-sig*(m.q.astype(ld)*x+m.c-m.A.T@y);p=np.minimum(np.maximum(raw,m.lo),m.hi)
    ax=m.A@x-m.rhs;v=ax-tau*y;s=np.zeros_like(v);s[:l]=np.clip(v[:l],m.al,m.au)
    vc=v[l:].reshape(-1,3);sc=s[l:].reshape(-1,3);t=vc[:,0];r=np.sqrt(np.sum(vc[:,1:]**2,axis=1));inside=r<=t;outside=(r<=-t)&~inside;mid=~(inside|outside)
    sc[inside]=vc[inside];sc[mid,0]=(r[mid]+t[mid])/2;sc[mid,1:]=((r[mid]+t[mid])/(2*r[mid]))[:,None]*vc[mid,1:]
    F=np.r_[x-p,ax-s];cert=native_replay(mp.ref,mp.lift(u))['certificate'];out=dict(F=F,norm=float(np.sqrt(np.sum(F*F))),certificate=cert)
    if direction is not None:
        dx=np.asarray(direction[:n],dtype=ld);dy=np.asarray(direction[n:],dtype=ld);draw=dx-sig*(m.q.astype(ld)*dx-m.A.T@dy)
        dp=((raw>m.lo)&(raw<m.hi))*draw;da=m.A@dx;dv=da-tau*dy;ds=np.zeros_like(dv);ds[:l]=((v[:l]>m.al)&(v[:l]<m.au))*dv[:l]
        dc=dv[l:].reshape(-1,3);dsc=ds[l:].reshape(-1,3);dsc[inside]=dc[inside]
        unit=vc[mid,1:]/r[mid,None];ud=np.sum(unit*dc[mid,1:],axis=1)
        dsc[mid,0]=(dc[mid,0]+ud)/2;dsc[mid,1:]=unit*(dc[mid,0]/2)[:,None]+((1+t[mid]/r[mid])/2)[:,None]*dc[mid,1:]-(t[mid]*ud/(2*r[mid]))[:,None]*unit
        out['Jd']=np.r_[dx-dp,da-ds]
    return out
