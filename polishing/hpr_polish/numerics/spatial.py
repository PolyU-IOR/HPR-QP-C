"""Structural numerical primitives; no experiment runner or standalone optimizer."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la
from scipy.sparse.csgraph import connected_components
from . import boundary as archived
from . import kkt
from .poisson import poisson
from .native import Native,projection,norm,maxabs


def state_indices(m):
    A=m.A[:m.linear_m];degree=np.diff(A.tocsc().indptr);cols=np.flatnonzero(degree>1)
    sub=A[:,cols].tocoo();pos=sub.data>0;I=np.empty(m.linear_m,int)
    assert len(cols)==m.linear_m and sum(pos)==m.linear_m
    assert len(np.unique(sub.row[pos]))==len(np.unique(sub.col[pos]))==m.linear_m
    I[sub.row[pos]]=cols[sub.col[pos]]
    return I

class Boundary(archived.SymmetricBoundary):
    def __init__(self,mp,state,grid):
        self.mp=mp;self.state=state;self.grid=grid;self.e=mp.evaluate(state)
        self.I=state_indices(mp.m);self.op=archived.BoundaryAction(mp,self.e,self.I,grid)
        self.B=self.op.B;self.nb=len(self.B);self.size=self.op.size
        self.P,self.H,self.Hi=archived.spectral(self.e['raw_y'][mp.m.linear_m:])
        self.actions=0;self.action_s=0.
    def rhs_lift(self,b):
        mp=self.mp;m=mp.m;o=self.op;bx=b[:mp.n];br=b[mp.n:]
        fixed=np.zeros(mp.n);active=self.e['box_cell']!=0;fixed[active]=bx[active]
        _,_,S=projection(m,self.e['raw_y'],True);G=sp.eye(mp.rows,format='csr')-S
        rf=bx/mp.sigma;rg=br-G@(m.A@fixed)
        ds0=poisson(rg[:o.l],self.grid);dl0=poisson(o.qs*ds0-rf[self.I],self.grid)
        f=rf[self.B]+o.L.T@dl0;h=rg[o.l:]-o.G@(o.C@ds0);y1=self.P@h/mp.tau
        rhs=np.r_[f+self.wt(y1),-self.Hi@h]
        def lift(z):
            u=z[:self.nb];yc=self.H@z[self.nb:]+y1
            ds1=poisson(o.L@u,self.grid);dl1=poisson(o.qs*ds1+o.C.T@yc,self.grid)
            dx=fixed.copy();dx[self.B]=u;dx[self.I]=ds0-ds1
            return np.r_[dx,dl0-dl1,yc]
        return rhs,lift
    def gauges(self):
        # Same algebra as S640, CSC slices avoid scanning every bulk row per component.
        o=self.op;assert np.all(o.qs>0)
        C=sp.vstack([sp.diags(np.sqrt(o.qb)),o.L,self.H@o.R],format='csc')
        graph=(abs(C).T@abs(C)).tocsr();nc,labels=connected_components(graph,directed=False)
        rr=[];cc=[];dd=[];count=0;largest=0
        for j in range(nc):
            ix=np.flatnonzero(labels==j);largest=max(largest,len(ix));assert len(ix)<=32
            ci=C[:,ix].tocoo();rows,inverse=np.unique(ci.row,return_inverse=True)
            a=np.zeros((len(rows),len(ix)));np.add.at(a,(inverse,ci.col),ci.data)
            Z=np.eye(len(ix)) if not len(rows) else la.null_space(a,rcond=32*max(a.shape)*np.finfo(float).eps)
            for col in Z.T:rr.extend(ix);cc.extend([count]*len(ix));dd.extend(col);count+=1
        Z=sp.csc_matrix((dd,(rr,cc)),shape=(self.size,count))
        defect=max((norm(self.action(Z[:,j].toarray().ravel())) for j in range(count)),default=0.)
        assert defect<=1e-11
        return Z,dict(count=count,action_defect=defect,largest_component=largest)

def direction(mp,u,grid):
    tick=time.perf_counter();s=Boundary(mp,u,grid);rhs,lift=s.rhs_lift(-s.e['F']);Z,ga=s.gauges()
    compat=norm(Z.T@rhs)/max(norm(rhs),1e-300);ga['relative_compatibility']=compat
    rec=dict(status='RANGE_REJECT',gauges=ga,boundary_order=s.size)
    if compat>1e-10:return None,rec
    M,pm=s.preconditioner(Z);rec.update(preconditioner=pm,setup_s=time.perf_counter()-tick)
    A=sla.LinearOperator((s.size,s.size),matvec=lambda v:s.action(v)+Z@(Z.T@v));hist=[]
    rr=rhs-Z@(Z.T@rhs);tick=time.perf_counter()
    z,info=sla.minres(A,rr,M=M,rtol=1e-12,maxiter=200,callback=lambda v:hist.append(None),check=False)
    d=lift(z);rec.update(iterations=len(hist),minres_info=info,linear_s=time.perf_counter()-tick,
        actions=s.actions,action_s=s.action_s,boundary_defect=norm(s.action(z)-rhs)/max(norm(rhs),1e-300))
    tick=time.perf_counter();ia=kkt.replay(mp,u,d)
    forcing=norm(ia['Jd']+ia['F'])/max(ia['norm'],1e-300)
    rec.update(independent_defect=forcing,check_s=time.perf_counter()-tick,direction_norm=norm(d),
               status='OK' if np.isfinite(forcing) and forcing<=1e-8 else 'LINEAR_REJECT')
    return d,rec
