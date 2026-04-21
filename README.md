# HPR-QP-C

C/CUDA implementation of the HPRQP solver for MPS-based quadratic programs.

## Install

For a fresh setup:

```sh
make clean && make
```

Install the shared library:

```sh
make install
```

Wrapper documentation:

- [Julia wrapper API](julia/README.md)
- [Python wrapper API](python/README.md)

## solve_qp

The executable is `solve_qp`.

GPU example:

```sh
./build-cuda/solve_qp model.mps --use-gpu 1 --device-number 0 --stoptol 1e-6 --time-limit 3600
```

Usage:

```sh
./build/solve_qp [mps_path] [--max-iter N] [--stoptol X] [--check-iter N] [--print-frequency N] [--time-limit X] [--verbose 0|1] [--use-gpu 0|1] [--device-number N] [--spmv-mode-q auto|cusparse|customized] [--spmv-mode-a auto|cusparse|customized] [--use-scaling 0|1]
```

CPU example:

```sh
./build/solve_qp model.mps --stoptol 1e-6 --time-limit 3600
```

Parameters:

- `mps_path`: input `.mps` file path; defaults to `model.mps`
- `--max-iter N`: maximum iteration count
- `--stoptol X`: stopping tolerance
- `--check-iter N`: interval for residual and restart checks
- `--print-frequency N`: iteration log interval; use `-1` for adaptive printing
- `--time-limit X`: time limit in seconds
- `--verbose 0|1`: enable or disable solver logging
- `--use-gpu 0|1`: enable or disable GPU solving
- `--device-number N`: CUDA device index when GPU solving is enabled
- `--spmv-mode-q auto|cusparse|customized`: SpMV backend for `Q`
- `--spmv-mode-a auto|cusparse|customized`: SpMV backend for `A` and `A^T`
- `--use-scaling 0|1`: enable or disable problem scaling

Output fields:

- `status`
- `iter`
- `time`
- `objective`
- `residual`
- `gap`
- `iter_4`
- `time_4`
- `iter_6`
- `time_6`