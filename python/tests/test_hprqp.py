from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

import hprqp


class HPRQPTests(unittest.TestCase):
    def test_status_name(self) -> None:
        self.assertEqual(hprqp.status_name(hprqp.STATUS_OPTIMAL), "OPTIMAL")

    def test_read_from_mps_and_max_iter(self) -> None:
        mps_path = Path(__file__).resolve().parents[2] / "tests" / "data" / "simple_qp.mps"
        model = hprqp.read_from_mps(mps_path)
        params = hprqp.Parameters(verbose=False, max_iter=0)
        result = hprqp.optimize(model, params)

        self.assertEqual(result.status_code, hprqp.STATUS_MAX_ITER)
        self.assertEqual(result.status, "MAX_ITER")
        self.assertEqual(len(result.x), 2)
        self.assertEqual(len(result.y), 2)

    def test_build_from_qabc(self) -> None:
        Q = [[2.0, 0.0], [0.0, 2.0]]
        A = []
        c = [-4.0, -2.0]
        AL = []
        AU = []
        l = [-math.inf, -math.inf]
        u = [math.inf, math.inf]
        model = hprqp.build_from_qabc(Q, c, A, AL, AU, l, u)
        params = hprqp.Parameters(verbose=False, stoptol=1e-8)
        result = hprqp.optimize(model, params)

        self.assertEqual(result.status_code, hprqp.STATUS_OPTIMAL)
        self.assertAlmostEqual(result.objective, -5.0, places=6)
        self.assertAlmostEqual(result.x[0], 2.0, places=6)
        self.assertAlmostEqual(result.x[1], 1.0, places=6)
        self.assertEqual(result.y, [])

    def test_gpu_path(self) -> None:
        if not hprqp.gpu_backend_available():
            self.skipTest("GPU backend unavailable")

        Q = [[2.0, 0.0], [0.0, 4.0]]
        A = [[1.0, 1.0]]
        c = [2.0, 1.0]
        AL = [2.0]
        AU = [2.0]
        l = [0.0, 0.0]
        u = [10.0, 10.0]
        model = hprqp.build_from_qabc(Q, c, A, AL, AU, l, u)
        cpu_result = hprqp.optimize(model, hprqp.Parameters(verbose=False, use_gpu=False))
        gpu_result = hprqp.optimize(model, hprqp.Parameters(verbose=False, use_gpu=True, device_number=0))

        self.assertEqual(cpu_result.status_code, hprqp.STATUS_OPTIMAL)
        self.assertEqual(gpu_result.status_code, hprqp.STATUS_OPTIMAL)
        self.assertAlmostEqual(cpu_result.objective, gpu_result.objective, places=6)
        self.assertAlmostEqual(cpu_result.residual, gpu_result.residual, places=6)
        self.assertAlmostEqual(cpu_result.gap, gpu_result.gap, places=6)
        self.assertAlmostEqual(cpu_result.x[0], gpu_result.x[0], places=6)
        self.assertAlmostEqual(cpu_result.x[1], gpu_result.x[1], places=6)


if __name__ == "__main__":
    unittest.main()