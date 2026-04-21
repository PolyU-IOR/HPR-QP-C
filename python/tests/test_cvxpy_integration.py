from __future__ import annotations

import unittest

import hprqp

try:
    import cvxpy as cp
except ImportError:
    cp = None


@unittest.skipUnless(cp is not None, "cvxpy not installed")
class HPRQPCVXPYTests(unittest.TestCase):
    @staticmethod
    def _make_problem() -> tuple[object, object, object]:
        import numpy as np

        x = cp.Variable(2, name="x")
        y = cp.Variable(name="y")
        problem = cp.Problem(
            cp.Minimize(
                0.5 * cp.quad_form(x, np.diag([2.0, 2.0]))
                - np.array([2.0, 4.0]) @ x
                + cp.square(y)
                - 6.0 * y
                + 21.0
            ),
            [x[0] + y == 4.0, x[1] >= 1.0, y >= 0.5],
        )
        return problem, x, y

    def test_build_from_cvxpy_problem(self) -> None:
        problem, x, y = self._make_problem()

        model, info = hprqp.build_from_cvxpy_problem(problem)

        self.assertEqual(model.num_variables, info.num_variables)
        self.assertEqual(model.num_constraints, info.num_constraints)
        self.assertEqual(info.num_variables, 3)
        self.assertEqual(info.num_constraints, 3)
        self.assertEqual(info.variable_offsets[x.id], 0)
        self.assertEqual(info.variable_offsets[y.id], 2)
        self.assertEqual(info.variable_shapes[x.id], (2,))
        self.assertEqual(info.variable_shapes[y.id], ())
        self.assertAlmostEqual(info.objective_constant, 21.0, places=6)

    def test_solve_cvxpy_problem_assigns_values(self) -> None:
        problem, x, y = self._make_problem()

        result = hprqp.solve_cvxpy_problem(problem, hprqp.Parameters(verbose=False, use_gpu=False))

        self.assertEqual(result.status_code, hprqp.STATUS_OPTIMAL)
        self.assertAlmostEqual(result.objective, 7.0, places=6)
        self.assertAlmostEqual(x.value[0], 1.0, places=6)
        self.assertAlmostEqual(x.value[1], 2.0, places=6)
        self.assertAlmostEqual(float(y.value), 3.0, places=6)

    def test_solve_cvxpy_problem_gpu_path(self) -> None:
        if not hprqp.gpu_backend_available():
            self.skipTest("GPU backend unavailable")

        import numpy as np

        x = cp.Variable(2, name="x")
        problem = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(x, np.array([[2.0, 0.0], [0.0, 4.0]])) + cp.sum(cp.multiply(np.array([2.0, 1.0]), x))),
            [x[0] + x[1] == 2.0, x >= 0.0],
        )

        cpu_result = hprqp.solve_cvxpy_problem(problem, hprqp.Parameters(verbose=False, use_gpu=False), assign_values=False)
        gpu_result = hprqp.solve_cvxpy_problem(
            problem,
            hprqp.Parameters(verbose=False, use_gpu=True, device_number=0),
            assign_values=False,
        )

        self.assertEqual(cpu_result.status_code, hprqp.STATUS_OPTIMAL)
        self.assertEqual(gpu_result.status_code, hprqp.STATUS_OPTIMAL)
        self.assertAlmostEqual(cpu_result.objective, gpu_result.objective, places=6)
        self.assertAlmostEqual(cpu_result.x[0], gpu_result.x[0], places=6)
        self.assertAlmostEqual(cpu_result.x[1], gpu_result.x[1], places=6)


if __name__ == "__main__":
    unittest.main()