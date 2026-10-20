import time
import numpy as np
import scipy.sparse as sp
from .access import AccessDecline,SparseAccess
from .band_factor import FixedBand


class BandAccess:
    def __init__(self,system):
        tick=time.monotonic();self.system=system;self.J=system.matrix().tocsr();m=system.m;e=system.e0
        identity=np.zeros(m.m,bool);l=m.linear_m
        identity[:l]=e['S'].diagonal==1
        for a,b,kind,_,_ in e['S'].blocks:
            identity[a:b]=(kind=='identity')
        delta=m.Q-sp.diags(m.Q.diagonal());delta.eliminate_zeros()
        gauges=np.empty(0,dtype=int)
        if not delta.nnz:
            forbidden=np.asarray(abs(m.A[~identity]).sum(axis=0)).ravel()
            gauges=np.flatnonzero((e['E']==1)&(m.Q.diagonal()==0)&(m.c==0)&(forbidden==0))
        self.gauges=gauges;K=self.J
        defects=dict(zero_columns=0.,dependent_rows=0.)
        if len(gauges):
            dep=self.J[gauges,:]+(system.alpha/system.beta)*(m.A[:,gauges].T@self.J[m.n:,:])
            coldef=float(np.max(abs(self.J[:,gauges].data),initial=0))
            rowdef=float(np.max(abs(dep.data),initial=0))
            defects=dict(zero_columns=coldef,dependent_rows=rowdef)
            if max(coldef,rowdef)>1e-11:raise AccessDecline('BAND_GAUGE_IDENTITY_REJECT')
            keep=np.ones(self.J.shape[0]);keep[gauges]=0
            K=sp.diags(keep)@self.J+sp.csr_matrix((np.ones(len(gauges)),(gauges,gauges)),shape=self.J.shape)
            K=K.tocsr();K.eliminate_zeros()
        self.factor=FixedBand(K)
        self.info=dict(access='common_kkt_band',**self.factor.info,gauges=len(gauges),
                       gauge_identity=defects,setup_total_s=time.monotonic()-tick)
    def solve(self,rhs):
        tick=time.monotonic();r=rhs.copy();r[self.gauges]=0
        d=self.factor.solve(r)
        defect=float(np.linalg.norm(self.system.J@d-rhs)/max(np.linalg.norm(rhs),1e-300))
        return d,dict(defect=defect,solve_s=time.monotonic()-tick)


def qualified_access(system):
    try:return BandAccess(system)
    except ValueError as ex:
        if 'BAND_FACTOR_REJECT' not in str(ex):raise
        if system.J.shape[0]>512:raise AccessDecline('UNRESOLVED_BAND_NULLSPACE:'+str(ex))
        fallback=SparseAccess(system)
        fallback.info['band_decline']=str(ex)
        return fallback
