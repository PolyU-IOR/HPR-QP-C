from __future__ import annotations

import sys
from pathlib import Path

import hprqp


def main() -> None:
    mps_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "tests" / "data" / "simple_qp.mps"

    print(f"Solving MPS: {mps_path}")
    print(f"GPU backend available: {hprqp.gpu_backend_available()}")

    model = hprqp.read_from_mps(mps_path)
    params = hprqp.Parameters(use_gpu=True, device_number=0, verbose=False, stoptol=1e-8)
    result = hprqp.optimize(model, params)

    print(f"status = {result.status}")
    print(f"objective = {result.objective}")
    print(f"residual = {result.residual}")
    print(f"gap = {result.gap}")
    print(f"iter = {result.iter}")
    print(f"x = {result.x}")
    print(f"y = {result.y}")


if __name__ == "__main__":
    main()