import unittest
import numpy as np
import scipy.sparse as sp
from hpr_polish.core import Model,Config
from hpr_polish.common_correction.episode import polish_common
from hpr_polish.cpu_check import check,prepare_candidate
from fixtures import qp

class Checker(unittest.TestCase):
    def test_general_q_correction(self):
        m,_=qp('affine');u=np.array([.3,.4,.01]);r=polish_common(m,u,Config())
        self.assertIsNotNone(r['candidate']);c=prepare_candidate(m,r['candidate']);self.assertTrue(check(m,c)['passed'])
        for key in ('x','w','z','y'):
            bad={k:v.copy() for k,v in c.items()};bad[key][0]+=1
            self.assertFalse(check(m,bad)['passed'],key)
        bad={k:v.copy() for k,v in c.items()};bad['s'][:]=123
        self.assertTrue(check(m,bad)['passed']) # cached slack is not trusted
    def test_false_paired_gap(self):
        m=Model([[0.]],sp.csr_matrix((0,1)),[1.],[0.],[1.],[],[],[])
        c=dict(x=np.array([1.]),w=np.array([1.]),z=np.array([1.]),y=np.empty(0),s=np.empty(0))
        r=check(m,c);self.assertFalse(r['passed']);self.assertEqual(r['relative_gap'],.5)
    def test_nonfinite_and_dual_domain(self):
        m=Model([[0.]],sp.csr_matrix((0,1)),[1.],[-np.inf],[np.inf],[],[],[])
        c=dict(x=np.zeros(1),w=np.zeros(1),z=np.ones(1),y=np.empty(0),s=np.empty(0))
        self.assertFalse(check(m,c)['passed'])
        c['x'][0]=np.nan
        with self.assertRaises(ValueError):check(m,c)
    def test_soc_dual_domain(self):
        m=Model(np.eye(3),sp.eye(3),np.zeros(3),np.full(3,-np.inf),np.full(3,np.inf),np.full(3,-np.inf),np.full(3,np.inf),np.zeros(3),((0,3),))
        c=dict(x=np.zeros(3),w=np.zeros(3),z=np.zeros(3),y=np.zeros(3),s=np.zeros(3))
        self.assertTrue(check(m,c)['passed']);c['y'][0]=-1
        self.assertFalse(check(m,c)['passed'])

if __name__=='__main__':unittest.main()
