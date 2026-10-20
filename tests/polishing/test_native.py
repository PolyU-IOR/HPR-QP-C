"""Independent native-port tests. NumPy/SciPy are test oracles only."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp
from fixtures import qp, spatial
from hpr_polish.core import Model
from hpr_polish.common_correction.access import System
from hpr_polish.cpu_check import check
from hpr_polish.transport import write_model, write_snapshot, write_candidate, read_candidate

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("HPRQP_NATIVE_BUILD", ROOT / "build-native-polish")).resolve()
WORKER = BUILD / "hprqp_polish_worker"
INSPECT = BUILD / "hprqp_polish_inspect"


class Native(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.count = 0
        # Neither the native worker nor its children can find a Python runtime.
        self.env = dict(os.environ, PATH="/nonexistent", OPENBLAS_NUM_THREADS="1",
                        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.env.pop("PYTHONPATH", None)

    def model(self, m, u):
        self.count += 1
        p = self.root / str(self.count)
        p.mkdir()
        write_model(p / "model.bin", m)
        write_snapshot(p / "snapshot.bin", m, u)
        return p

    def inspect(self, p, mode="system", extra=(), success=True):
        inp = p / ("candidate.bin" if mode == "check" else "snapshot.bin")
        r = subprocess.run([str(INSPECT), mode, str(p / "model.bin"), str(inp), *extra],
                           env=self.env, capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode == 0, success, r.stderr)
        return json.loads(r.stdout) if success else None

    def worker(self, p, m, change=None, success=True):
        req = dict(token="ab" * 16, parent_pid=os.getpid(),
                   model=str(p / "model.bin"), seconds=20, memory_bytes=8 * 2**30,
                   dimensions=[m.n, m.m], iteration=100, tolerance=1e-8,
                   config=dict(backend="auto"))
        req.update(change or {})
        (p / "request.json").write_text(json.dumps(req))
        r = subprocess.run([str(WORKER), str(p / "request.json")], env=self.env,
                           capture_output=True, text=True, timeout=25)
        self.assertEqual(r.returncode == 0, success, r.stderr)
        return json.loads((p / "result.json").read_text()) if success else None

    def test_analytic_equations_and_certificates(self):
        for kind in ("affine", "mixed", "dependent", "no_equalities", "box"):
            with self.subTest(kind=kind):
                m, objective = qp(kind)
                u = np.r_[np.array([.25, .5]) + 1e-4, np.zeros(m.m)]
                p = self.model(m, u)
                r = self.inspect(p, extra=("sparse",))
                s = System(m, u.astype(np.longdouble), 1.)
                np.testing.assert_allclose(r["F"], s.e0["F"], atol=2e-16, rtol=1e-12)
                np.testing.assert_allclose(r["J"], s.matrix().toarray(), atol=1e-14)
                d = np.linspace(-.7, .9, len(u))
                np.testing.assert_allclose(r["Jd"], s.J @ d, atol=1e-14)
                self.assertLess(r["linear"]["defect"], 1e-12)
                result = self.worker(p, m)
                self.assertEqual(result["status"], "CPU_CERTIFIED", result)
                self.assertTrue((p / "READY").exists())
                self.assertEqual((p / "certificate.bin").stat().st_size, 120)
                cert = check(m, read_candidate(p / "candidate.bin", m))
                self.assertTrue(cert["passed"], cert)
                self.assertAlmostEqual(cert["primal_objective"], objective, places=9)
                self.assertEqual(result["active_set_updates"], 0)
                self.assertEqual(result["derivative_refreshes"], 0)

    def test_soc_derivatives_and_banded_equations(self):
        m = Model(sp.eye(3), sp.eye(3), np.zeros(3), np.full(3, -np.inf),
                  np.full(3, np.inf), np.full(3, -np.inf), np.full(3, np.inf),
                  np.zeros(3), ((0, 3),))
        for v in ([0., 1., 2.], [3., 1., 1.], [-3., 1., 1.], [0., 0., 0.]):
            with self.subTest(v=v):
                x = np.array([.2, -.1, .3])
                u = np.r_[x, x - v]
                p = self.model(m, u)
                r = self.inspect(p, extra=("band",))
                s = System(m, u.astype(np.longdouble), 1.)
                np.testing.assert_allclose(r["F"], s.e0["F"], atol=2e-15)
                np.testing.assert_allclose(r["J"], s.matrix().toarray(), atol=2e-15)
                self.assertLess(r["linear"]["defect"], 1e-12)
                if np.linalg.norm(v):
                    d = np.linspace(-.7, .9, len(u))
                    h = 1e-6
                    fd = (s.evaluate(u+h*d)["F"]-s.evaluate(u-h*d)["F"])/(2*h)
                    np.testing.assert_allclose(r["Jd"], fd, atol=1e-9)

    def test_independent_checker_rejects_corruption(self):
        m, _ = qp("affine")
        p = self.model(m, np.array([.3, .4, .01]))
        self.worker(p, m)
        c = read_candidate(p / "candidate.bin", m)
        for key in ("x", "w", "z", "y", "s"):
            with self.subTest(key=key):
                b = {k: v.copy() for k, v in c.items()}
                b[key][0] += 1.
                (p / "candidate.bin").unlink()
                write_candidate(p / "candidate.bin", m, b)
                r = self.inspect(p, "check")
                self.assertEqual(r["passed"], check(m, b)["passed"])
                self.assertEqual(r["passed"], key == "s")  # checker recomputes slack
        b["x"][0] = np.nan
        (p / "candidate.bin").unlink()
        write_candidate(p / "candidate.bin", m, b)
        self.inspect(p, "check", success=False)

    def test_spatial_access_solves_same_frozen_equations(self):
        for g in (3, 7):
            with self.subTest(grid=g):
                m = spatial(g)
                k = 4*g
                x = np.r_[np.full(k, .5), np.full(g*g, .7), np.full(k, 4.)]
                u = np.r_[x, np.zeros(m.m)]
                p = self.model(m, u)
                r = self.inspect(p, extra=("spatial",))
                s = System(m, u.astype(np.longdouble), 1.)
                np.testing.assert_allclose(r["J"], s.matrix().toarray(), atol=2e-14)
                d = np.linspace(-.7, .9, len(u))
                b = s.J @ d
                defect = np.linalg.norm(s.J @ np.array(r["solution"])-b)/np.linalg.norm(b)
                self.assertLess(defect, 1e-8)
                self.assertLess(r["linear"]["defect"], 1e-8)

    def test_dual_support_and_soc_domain(self):
        models = [
            Model([[0.]], sp.csr_matrix((0, 1)), [1.], [0.], [1.], [], [], []),
            Model([[0.]], sp.csr_matrix((0, 1)), [1.], [-np.inf], [np.inf], [], [], []),
            Model(sp.eye(3), sp.eye(3), np.zeros(3), np.full(3, -np.inf),
                  np.full(3, np.inf), np.full(3, -np.inf), np.full(3, np.inf),
                  np.zeros(3), ((0, 3),)),
        ]
        for m in models:
            p = self.model(m, np.zeros(m.n + m.m))
            c = dict(x=np.ones(m.n), w=np.ones(m.n), z=np.ones(m.n),
                     y=np.full(m.m, -1.), s=np.zeros(m.m))
            write_candidate(p / "candidate.bin", m, c)
            self.assertFalse(self.inspect(p, "check")["passed"])
            self.assertFalse(check(m, c)["passed"])

    def test_protocol_rejections_do_not_publish(self):
        m, _ = qp("affine")
        for change in (dict(parent_pid=1), dict(model_sha256="0"*64),
                       dict(config={"unexpected": 1}), dict(dimensions=[8, 9])):
            p = self.model(m, np.array([.3, .4, .01]))
            self.worker(p, m, change, success=False)
            self.assertFalse((p / "READY").exists())
        p = self.model(m, np.array([.3, .4, .01]))
        with (p / "model.bin").open("ab") as f:
            f.write(b"trailing corruption")
        self.worker(p, m, success=False)
        self.assertFalse((p / "READY").exists())


if __name__ == "__main__":
    unittest.main()
