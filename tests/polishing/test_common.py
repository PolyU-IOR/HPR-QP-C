"""Common correction: native certificates and frozen equation access."""
import unittest
import numpy as np
from fixtures import qp
from hpr_polish.core import Config
from hpr_polish.common_correction.access import System, SparseAccess
from hpr_polish.common_correction.episode import polish_common
from hpr_polish.cpu_check import check

class Common(unittest.TestCase):
    def test_analytic_qps(self):
        for kind in ('affine','mixed','dependent','no_equalities','box'):
            with self.subTest(kind=kind):
                m,opt=qp(kind);u=np.r_[np.array([.25,.5])+1e-4,np.zeros(m.m)]
                s=System(m,u,1.);a=SparseAccess(s);d,lin=a.solve(-s.e0['F'])
                self.assertLess(lin['defect'],1e-10)
                direction=np.linspace(-.7,.9,len(u));h=1e-6
                fd=(s.evaluate(u+h*direction)['F']-s.evaluate(u-h*direction)['F'])/(2*h)
                np.testing.assert_allclose(s.J@direction,fd,rtol=1e-7,atol=1e-9)
                r=polish_common(m,u,Config(tolerance=1e-8,seconds=10))
                self.assertTrue(r['certified'],r)
                self.assertEqual(r['correction'],'S734_common_projected_KKT')
                self.assertEqual(r['derivative_refreshes'],0)
                self.assertEqual(r['active_set_updates'],0)
                self.assertFalse(r['partial_feedback'])
                cert=check(m,r['candidate'],1e-8)
                self.assertTrue(cert['passed'],cert)
                self.assertAlmostEqual(cert['primal_objective'],opt,places=8)

if __name__=='__main__':unittest.main()
