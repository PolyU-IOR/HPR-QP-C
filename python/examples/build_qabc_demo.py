from __future__ import annotations

import hprqp


def main() -> None:
    Q = [[2.0, 0.0], [0.0, 4.0]]
    A = [[1.0, 1.0]]
    c = [2.0, 1.0]
    AL = [2.0]
    AU = [2.0]
    l = [0.0, 0.0]
    u = [10.0, 10.0]

    print(f"GPU backend available: {hprqp.gpu_backend_available()}")
    model = hprqp.build_from_qabc(Q, c, A, AL, AU, l, u)
    params = hprqp.Parameters(use_gpu=True, device_number=0, verbose=False, stoptol=1e-8)
    result = hprqp.optimize(model, params)

    print(f"status = {result.status}")
    print(f"objective = {result.objective}")
    print(f"x = {result.x}")
    print(f"y = {result.y}")


if __name__ == "__main__":
    main()