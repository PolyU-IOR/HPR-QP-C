"""Structural numerical primitives; no experiment runner or standalone optimizer."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la
from scipy.sparse.csgraph import connected_components
from .boundary_action import BoundaryAction
from .poisson import poisson
from .native import projection,norm,maxabs


def blockdiag(blocks):
    k=len(blocks);rr=3*np.arange(k)[:,None,None]+np.arange(3)[None,:,None]+np.zeros((1,1,3),int)
    cc=3*np.arange(k)[:,None,None]+np.arange(3)[None,None,:]+np.zeros((1,3,1),int)
    out=sp.csr_matrix((blocks.ravel(),(rr.ravel(),cc.ravel())),shape=(3*k,3*k));out.eliminate_zeros();return out

def spectral(raw):
    """Analytic spectral projectors; no threshold-based deletion of middle eigenvalues."""
    v=raw.reshape(-1,3);t=v[:,0];r=np.linalg.norm(v[:,1:],axis=1)
    inside=r<=t;outside=(r<=-t)&~inside;mid=~(inside|outside)
    P=np.zeros((len(v),3,3));H=P.copy();Hi=P.copy()
    P[inside]=np.eye(3);H[outside]=np.eye(3);Hi[outside]=np.eye(3)
    if np.any(mid):
        u=v[mid,1:]/r[mid,None];ep=np.column_stack([np.ones(sum(mid)),u])/np.sqrt(2)
        em=np.column_stack([np.ones(sum(mid)),-u])/np.sqrt(2)
        et=np.column_stack([np.zeros(sum(mid)),-u[:,1],u[:,0]])
        pp=ep[:,:,None]*ep[:,None,:];pm=em[:,:,None]*em[:,None,:];pt=et[:,:,None]*et[:,None,:]
        g=(r[mid]-t[mid])/(2*r[mid]);assert np.all(g>0)
        P[mid]=pp;H[mid]=pm+np.sqrt(g)[:,None,None]*pt;Hi[mid]=pm+1/np.sqrt(g)[:,None,None]*pt
    return blockdiag(P),blockdiag(H),blockdiag(Hi)

class SymmetricBoundary:
    def __init__(self,mp,state,grid):
        self.mp=mp;self.state=state;self.grid=grid;self.e=mp.evaluate(state)
        I=np.arange(4*grid,4*grid+grid*grid)
        self.op=BoundaryAction(mp,self.e,I,grid);self.I=I;self.B=self.op.B
        self.nb=len(self.B);self.size=self.op.size
        self.P,self.H,self.Hi=spectral(self.e['raw_y'][mp.m.linear_m:])
        self.actions=0;self.action_s=0.
    def wt(self,y):
        o=self.op;return o.R.T@y-o.L.T@poisson(o.C.T@y,self.grid)
    def action(self,z):
        tick=time.perf_counter();o=self.op;u=z[:self.nb];yy=z[self.nb:];yc=self.H@yy
        ds=poisson(o.L@u,self.grid);dl=poisson(o.qs*ds+o.C.T@yc,self.grid)
        out=np.r_[o.qb*u-o.R.T@yc+o.L.T@dl,-self.H@(o.R@u-o.C@ds)-self.mp.tau*(o.S@yy)]
        self.actions+=1;self.action_s+=time.perf_counter()-tick;return out
    def rhs_lift(self,b):
        mp=self.mp;m=mp.m;n=mp.n;sig=mp.sigma;o=self.op;bx,bw,by=np.split(b,[n,2*n]);aty=m.A.T@by
        gw=-bx-sig*aty/2+(1+sig*m.q)*bw/2
        gt=-bx+sig*m.q*bw/2-sig*sig*m.q*mp.d*aty/2
        fixed=np.zeros(n);active=self.e['box_cell']!=0;fixed[active]=bx[active]/2
        _,_,S=projection(m,self.e['raw_y'],True);G=sp.eye(mp.rows,format='csr')-S
        rf=bx/(2*sig)-m.q*gw;rg=mp.tau*by/2-G@(m.A@(gt+fixed))
        ds0=poisson(rg[:o.l],self.grid);dl0=poisson(o.qs*ds0-rf[self.I],self.grid)
        f=rf[self.B]+o.L.T@dl0;h=rg[o.l:]-o.G@(o.C@ds0);y1=self.P@h/mp.tau
        rhs=np.r_[f+self.wt(y1),-self.Hi@h]
        def lift(z):
            u=z[:self.nb];yc=self.H@z[self.nb:]+y1
            ds1=poisson(o.L@u,self.grid);dl1=poisson(o.qs*ds1+o.C.T@yc,self.grid)
            dx=fixed.copy();dx[self.B]=u;dx[self.I]=ds0-ds1;dy=np.r_[dl0-dl1,yc]
            return np.r_[dx,dx+gw,dy]
        return rhs,lift
    def gauges(self):
        o=self.op;assert np.all(o.qs>0), 'primal gauge identification requires positive state Q'
        # Qred h=0 iff qb*h=0 and L*h=0 under this explicit assumption.
        C=sp.vstack([sp.diags(np.sqrt(o.qb)),o.L,self.H@o.R],format='csr')
        graph=(abs(C).T@abs(C)).tocsr();nc,labels=connected_components(graph,directed=False)
        rr=[];cc=[];dd=[];count=0;largest=0
        for j in range(nc):
            ix=np.flatnonzero(labels==j);largest=max(largest,len(ix));assert len(ix)<=32,'gauge local component too large'
            ci=C[:,ix];rows=np.flatnonzero(np.asarray(abs(ci).sum(axis=1)).ravel());a=ci[rows].toarray()
            if not len(rows):Z=np.eye(len(ix))
            else:Z=la.null_space(a,rcond=32*max(a.shape)*np.finfo(float).eps)
            for col in Z.T:
                rr.extend(ix);cc.extend([count]*len(ix));dd.extend(col);count+=1
        Z=sp.csc_matrix((dd,(rr,cc)),shape=(self.size,count))
        defect=max((norm(self.action(Z[:,j].toarray().ravel())) for j in range(count)),default=0.)
        if defect>1e-11:raise RuntimeError(f'gauge identity defect {defect}')
        return Z,dict(count=count,action_defect=defect,largest_component=largest)
    def preconditioner(self,Z):
        o=self.op;invdiag=1/o.K.diagonal();T=sp.diags(invdiag)@o.L
        W=o.R-o.C@T;Q=sp.diags(o.qb)+T.T@sp.diags(o.qs)@T
        H=sp.bmat([[Q,-W.T@self.H],[-self.H@W,-self.mp.tau*o.S]],format='csr');H.eliminate_zeros()
        nc,lab=connected_components(abs(H)+abs(H.T),directed=False)
        rr=[];cc=[];data=[];floored=0;sizes=[]
        for j in range(nc):
            ix=np.flatnonzero(lab==j);sizes.append(len(ix));assert len(ix)<=32,'preconditioner block too large'
            a=H[ix][:,ix].toarray();vals,U=la.eigh((a+a.T)/2)
            cut=max(1e-12,1e-3*maxabs(vals));floored+=sum(abs(vals)<cut)
            P=(U/np.maximum(abs(vals),cut))@U.T
            rr.extend(np.repeat(ix,len(ix)));cc.extend(np.tile(ix,len(ix)));data.extend(P.ravel())
        P=sp.csr_matrix((data,(rr,cc)),shape=H.shape)
        def proj(v):return v-Z@(Z.T@v)
        def action(v):return proj(P@proj(v))+Z@(Z.T@v)
        return sla.LinearOperator(H.shape,matvec=action),dict(blocks=nc,largest_block=max(sizes),floored_eigenvalues=floored,nnz=P.nnz)
