# HPR-QP-C

A CUDA/C implementation of the Halpern Peaceman--Rachford (HPR) method for solving quadratic programming (QP) problems on GPUs.

This branch packages the validated updates as a single commit on GitHub `main`.
See the [clean release scope](docs/CLEAN_RELEASE_SCOPE.md) and
[qualification record](docs/INTEGRATION_VALIDATION.md).

## SOC support

This branch adds GPU standard second-order cones through the C array API.
See [SOC usage and scope](docs/SOC.md) and [the C example](examples/solve_socp.c).
Existing QP entry points and solver policies are retained.
The [low-rank QCQP bridge](include/shared/qcqp.h) retains sparse quadratic factors.
The experimental Julia policy switches are not included in this production snapshot.

## Simplified CPU polishing and equality Gram

The common frozen projected-KKT correction uses specialized linear algebra only;
all backends share residuals, continuation and the independent native certificate.
Independent GPU-owner controls provide terminal-only CPU polishing and optional
equality Gram scaling. CPU polishing is enabled by default; Gram defaults to
`auto` for conservative structural selection. Use `--polishing 0 --gram off`
to disable both, or use the corresponding C API options.
See [configuration, contracts and tests](docs/POLISHING_AND_GRAM.md) and the
[formal equality-scaling note](docs/equality_gram_scaling.tex).

Direct executable and dataset invocations enable these defaults without extra
flags. Install the worker dependencies in `python3` on PATH and cuDSS for Gram.
Unsupported models or missing optional CUDA capabilities retain the documented
fallbacks. `make run` and `make dataset` prefer the GPU build when CUDA is present.

```sh
python3 scripts/run_dataset.py /path/to/mps /path/to/results
```

## Install

For a fresh setup:

```sh
python3 -m pip install -r polishing/requirements.txt
make
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
./build-cuda/solve_qp model.mps
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

### Automatic cuSPARSE SpMVOp

CUDA Toolkit 13.3 or newer builds automatically use FP64 SpMVOp ALG1 for
cuSPARSE products with A, A-transpose and Q. Older toolkit builds retain
preprocessed CSR ALG2. If SpMVOp plan setup is unsupported or cannot allocate
its workspace, setup falls back to CSR ALG2; execution errors are reported.
Verbose output records each operator's actual backend and scratch size.
Existing customized kernels and zero/diagonal-Q handling are unchanged.
Gram and CPU polishing retain their existing defaults and numerical rules.

Use a driver compatible with the selected toolkit (or NVIDIA's supported
forward-compatibility package). The backend does not install or modify drivers.
The focused `test_gpu_spmv_operation` test covers changed vector descriptors,
CUDA graph capture, plan replacement, and setup-failure fallback.

### Packed sparse iteration updates

The GPU solver automatically attempts packed A/AT access for interval-only models
when both matrices have at most 32 entries per row, padding uses at most 1.5
times the CSR entries, and the pair needs at most half of currently free GPU
memory. It preserves FP64 coefficients and CSR row order, and refreshes the
packed values after scaling. The existing y/w updates fuse these products with
their vector operations; HPR, Halpern and polishing rules are unchanged.

Unsupported shapes, SOC models and allocation failures retain the existing
product path. CSR remains authoritative for residuals, controllers and other
products, including SpMVOp where supported. The packed iteration path may take
precedence over the requested A SpMV mode; set `HPRQP_PACKED_SPMV=0` to disable
it. Gram uses its existing update path. Preparation, memory and fallback costs
are included in solver timing.
