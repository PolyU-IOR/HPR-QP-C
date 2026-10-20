"""Fixed banded factorization: linear access only, no correction policy."""
import time
import numpy as np
import scipy.linalg as la
from scipy.sparse.csgraph import reverse_cuthill_mckee


class FixedBand:
    def __init__(self, matrix):
        tick = time.monotonic()
        self.matrix = matrix.tocsr()
        p = self.matrix.copy(); p.data[:] = 1
        self.perm = reverse_cuthill_mckee((p+p.T).tocsr(), symmetric_mode=True)
        a = self.matrix[self.perm][:, self.perm].tocoo()
        self.lo = int(np.max(a.row-a.col, initial=0))
        self.hi = int(np.max(a.col-a.row, initial=0))
        memory = 8*(2*self.lo+self.hi+1)*matrix.shape[0]
        if max(self.lo,self.hi)>128 or memory>2**30:
            raise ValueError('BAND_RESOURCE_SKIP')
        ab = np.zeros((2*self.lo+self.hi+1,matrix.shape[0]), order='F')
        ab[self.lo+self.hi+a.row-a.col,a.col] = a.data
        factor, self.solve_lapack = la.lapack.get_lapack_funcs(('gbtrf','gbtrs'), (ab,))
        self.lu,self.piv,info = factor(ab,self.lo,self.hi,overwrite_ab=True)
        if info != 0:
            raise ValueError('BAND_FACTOR_REJECT:'+str(info))
        self.info = dict(factorizations=1,jacobians=1,dimension=matrix.shape[0],
                         lower_bandwidth=self.lo,upper_bandwidth=self.hi,
                         factor_bytes=memory,factor_s=time.monotonic()-tick)

    def solve(self,rhs):
        b = np.array(rhs[self.perm,None],order='F')
        z,info = self.solve_lapack(self.lu,self.lo,self.hi,b,self.piv,overwrite_b=True)
        if info != 0: raise ValueError('BAND_SOLVE_REJECT:'+str(info))
        answer = np.empty_like(rhs); answer[self.perm] = z[:,0]
        return answer
