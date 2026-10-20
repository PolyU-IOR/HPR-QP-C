"""Fixed equation access. No nonlinear continuation or publication decisions."""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from hpr_polish import core
from hpr_polish.adapters import as_legacy
from hpr_polish.numerics import native, kkt, spatial


class AccessDecline(RuntimeError):
    pass


def beta_rule(m, name):
    if name == 'unit': return 1.
    if name != 'norm': raise ValueError(name)
    a=float(np.max(np.asarray(abs(m.A).sum(axis=0)),initial=0))
    b=float(np.max(np.asarray(abs(m.A).sum(axis=1)),initial=0))
    return 1.1*a*b if a*b>0 else 1.


class System:
    def __init__(self,m,u,beta):
        self.m=m;self.u0=u.copy();self.alpha=1.;self.beta=beta
        self.cfg=core.Config(alpha=1.,beta=beta,direct_dimension=3_000_000,
                             assembly_nnz=40_000_000,dense_cone_limit=64)
        self.e0=self.evaluate(u)
        self.J=core.jacobian(m,self.e0,1.,beta)
    def evaluate(self,u): return core.evaluate(self.m,u,1.,self.beta)
    def candidate(self,u,e):return core.candidate(self.m,u,e,1.,self.beta)
    def matrix(self):
        j=core.sparse_jacobian(self.m,self.e0,self.cfg)
        if j is None:raise AccessDecline('ASSEMBLY_ACCESS_RESOURCE')
        return j


class SparseAccess:
    def __init__(self,system):
        tick=time.monotonic();self.system=system;self.J=system.matrix();self.calls=0
        if self.J.shape[0]<=512:
            a=self.J.toarray();self.U,self.s,self.Vh=la.svd(a,full_matrices=False)
            self.rank=int(np.count_nonzero(self.s>max(a.shape)*np.finfo(float).eps*self.s[0])) if len(self.s) and self.s[0] else 0
            self.factor=None;kind='dense_svd'
        else:
            self.factor=sla.splu(self.J);kind='sparse_lu';self.rank=None
        self.info=dict(access=kind,dimension=self.J.shape[0],setup_s=time.monotonic()-tick,
                       factorizations=1,rank=self.rank)
    def solve(self,rhs):
        tick=time.monotonic();self.calls+=1
        if self.factor is None:
            r=self.rank;d=self.Vh[:r].T@((self.U[:,:r].T@rhs)/self.s[:r])
        else:d=self.factor.solve(rhs)
        defect=np.linalg.norm(self.system.J@d-rhs)/max(np.linalg.norm(rhs),1e-300)
        return d,dict(defect=float(defect),solve_s=time.monotonic()-tick)


class FrozenSpatial:
    """Extracted boundary factor/preconditioner, reused for arbitrary RHS.

    canonical=False preserves the historical reduced-gauge choice for reference
    extraction tests. canonical=True removes any explicit full-space null
    component so generic and spatial use the same Euclidean convention.
    """
    def __init__(self,mp,u,grid,canonical=False):
        tick=time.monotonic();self.mp=mp;self.u=u.copy();self.calls=0
        self.s=spatial.Boundary(mp,u,grid);self.Z,self.ga=self.s.gauges()
        self.M=None;self.pm=None;self.canonical=canonical;self.G=None
        if canonical and self.Z.shape[1]:
            count=self.Z.shape[1];bytes_needed=8*mp.size*count
            if bytes_needed>8*1024**3:raise AccessDecline('NULLSPACE_ACCESS_RESOURCE')
            _,lift0=self.s.rhs_lift(np.zeros(mp.size))
            self.G=np.column_stack([lift0(self.Z[:,i].toarray().ravel()) for i in range(count)])
            self.GU,self.Gs,self.GVh=la.svd(self.G,full_matrices=False)
            assert np.min(self.Gs)>np.max(self.Gs)*max(self.G.shape)*np.finfo(float).eps
        self.info=dict(access='spatial',boundary_order=self.s.size,gauges=self.ga,
                       setup_s=time.monotonic()-tick,factorizations=0,preconditioners=0)
    def solve(self,b):
        tick=time.monotonic();self.calls+=1;s=self.s;Z=self.Z
        rhs,lift=s.rhs_lift(b)
        compat=np.linalg.norm(Z.T@rhs)/max(np.linalg.norm(rhs),1e-300)
        if compat>1e-10:raise AccessDecline('SPATIAL_RANGE_REJECT:'+str(compat))
        if self.M is None:
            self.M,self.pm=s.preconditioner(Z);self.info['preconditioners']=1
        A=sla.LinearOperator((s.size,s.size),matvec=lambda v:s.action(v)+Z@(Z.T@v))
        hist=[];rr=rhs-Z@(Z.T@rhs)
        z,flag=sla.minres(A,rr,M=self.M,rtol=1e-12,maxiter=200,
                         callback=lambda _:hist.append(None),check=False)
        d=lift(z)
        if self.G is not None:d=d-self.GU@(self.GU.T@d)
        # Frozen original-coordinate action; RHS need not be the initial F.
        ia=kkt.replay(self.mp,self.u,d)
        defect=np.linalg.norm(ia['Jd']-b)/max(np.linalg.norm(b),1e-300)
        return d,dict(defect=float(defect),relative_compatibility=float(compat),
            iterations=len(hist),minres_info=int(flag),solve_s=time.monotonic()-tick,
            boundary_defect=float(np.linalg.norm(s.action(z)-rhs)/max(np.linalg.norm(rhs),1e-300)),
            preconditioner=self.pm)


class SpatialAccess:
    def __init__(self,system):
        m=system.m
        extra=m.Q-core.sp.diags(m.Q.diagonal());extra.eliminate_zeros()
        if extra.nnz or any(b-a!=3 for a,b in m.cones):raise AccessDecline('SPATIAL_MODEL_UNSUPPORTED')
        old=as_legacy(m,native);ref=native.Native(old);ref.tau=system.beta
        mp=kkt.KKT(ref);e=mp.evaluate(system.u0)
        if np.any(np.all(e['raw_y'][old.linear_m:].reshape(-1,3)==0,axis=1)):
            raise AccessDecline('SPATIAL_APEX_DERIVATIVE_UNSUPPORTED')
        I=spatial.state_indices(old)
        if np.any(e['box_cell'][I]):raise AccessDecline('SPATIAL_STATE_PIVOT_UNAVAILABLE')
        grid=int(round(np.sqrt(old.linear_m)))
        if grid*grid!=old.linear_m:raise AccessDecline('SPATIAL_TENSOR_UNAVAILABLE')
        self.system=system;self.inner=FrozenSpatial(mp,system.u0,grid,canonical=True)
        self.mp=mp;self.info=self.inner.info
    def solve(self,rhs):
        d,rec=self.inner.solve(rhs)
        rec['legacy_original_defect']=rec['defect']
        rec['defect']=float(np.linalg.norm(self.system.J@d-rhs)/max(np.linalg.norm(rhs),1e-300))
        return d,rec

