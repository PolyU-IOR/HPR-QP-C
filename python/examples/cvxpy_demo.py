from __future__ import annotations

import cvxpy as cp
import numpy as np

import hprqp


def main() -> None:
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

    use_gpu = hprqp.gpu_backend_available()
    params = hprqp.Parameters(verbose=False, use_gpu=use_gpu, device_number=0)
    result = hprqp.solve_cvxpy_problem(problem, params)

    print(f"backend={'gpu' if use_gpu else 'cpu'}")
    print(f"status={result.status}")
    print(f"objective={result.objective:.6f}")
    print(f"x={x.value}")
    print(f"y={float(y.value):.6f}")


if __name__ == "__main__":
    main()