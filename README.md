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
The executable and dataset runner also load diagonal convex `QCMATRIX`
constraints directly from MPS files; see [supported input and limitations](docs/SOC.md#qcqp-mps-input).
Unsupported quadratic input is rejected rather than silently dropped.
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
flags. The CPU polishing worker is native C++; NumPy and SciPy are not runtime
dependencies. The GPU build supplies the separate cuDSS dependency for Gram.
Unsupported models retain the documented fallbacks. A default GPU build requires
cuDSS, so it cannot silently omit Gram. `make run` and `make dataset` prefer the
GPU build when CUDA is present.

```sh
python3 scripts/run_dataset.py /path/to/mps /path/to/results
```

## Install

For a fresh setup, provide a C/C++ compiler, Python 3 and OpenSSL `libcrypto`
(3 or 1.1). Python is used by build helpers and the optional dataset script,
not by polishing. On Debian/Ubuntu Linux x86_64, `make` also installs missing
CMake and BLAS/LAPACK prerequisites into this checkout automatically.
With a compatible CUDA toolkit and driver installed:

```sh
make
```

For native CPU polishing, `make` first reuses working CMake and BLAS/LAPACK
installations. If missing, it downloads checksum-verified CMake 3.28.4 and/or
extracts the distribution's BLAS/LAPACK and Fortran runtime packages under
`.deps/`. Package versions and hashes come from the configured APT repository
metadata. This requires `apt-get`/`dpkg-deb` with usable package indexes for the
native libraries. It does not run `sudo`, execute package installation scripts,
or modify system packages. `make setup-native` can prepare this explicitly.
`AUTO_INSTALL_NATIVE_DEPS=0` disables these downloads and reports missing
dependencies. On other platforms, install CMake 3.16+ and BLAS/LAPACK manually.

The native worker finds its private runtime libraries automatically; no
`LD_LIBRARY_PATH` setting is needed for these libraries. If you previously saw
“Native CPU polishing requires BLAS and LAPACK”, use this updated source and
run `make` again; the failed configuration is repaired without changing solver
settings.

On a GPU build, `make` reuses a compatible cuDSS installation or automatically
downloads the benchmark-qualified NVIDIA cuDSS
0.7.1.4 development archive matching the selected CUDA major (12 or 13), verifies
its published SHA256, installs it under `.deps/cudss/` in this checkout, and
continues the build with Gram enabled.
It does not change drivers, system packages or Python packages. NVIDIA's license
ships inside the archive. Linux x86_64 and SBSA are supported.
An initial download is about 110–120 MiB. `make AUTO_INSTALL_CUDSS=0` disables
network installation and fails clearly if the required library is missing.
CPU-only builds, help, cleaning and capability inspection do not download it.
`make setup-cudss` remains available to prepare the dependency separately.

If several CUDA toolkits are installed, use the same `CUDA_ROOT` in both commands:

```sh
make CUDA_ROOT=/usr/local/cuda-13.3
make check-cudss CUDA_ROOT=/usr/local/cuda-13.3
```

The selected toolkit also needs a compatible NVIDIA driver. On a server where
an administrator has installed CUDA's forward-compatibility libraries for a
supported GPU, enable that installation before running the solver, for example:

```sh
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

This applies to the CUDA driver, not to the private BLAS/LAPACK libraries.
If no compatible driver/compatibility package is available, choose a toolkit
supported by the installed driver (for example `make CUDA_ROOT=/usr/local/cuda-12.8`)
or ask the administrator to update the driver. `make` never modifies drivers.

Already have cuDSS? Use `make CUDSS_ROOT=/path/to/cudss`;
that directory must contain `include/cudss.h` and its matching shared library.
The build also detects NVIDIA wheels, including `libcudss.so.0` without a symlink.
Do not delete or move the cuDSS installation while using a binary linked to it.
See [NVIDIA's installation instructions](https://docs.nvidia.com/cuda/cudss/latest/getting_started.html).

For an older checkout previously built without cuDSS, run `make gpu-clean` before
rebuilding. The updated Makefile automatically rebuilds GPU objects when the
detected cuDSS capability or CUDA toolkit changes. `make REQUIRE_CUDSS=0` permits
an intentional GPU build without Gram; it will not reproduce Gram-enabled
experiments. CMake has the corresponding `HPRQP_REQUIRE_CUDSS=OFF` opt-out.
CPU-only builds (`make cpu`) do not require cuDSS.

The native worker is built and placed beside `build-cuda/solve_qp`.
Keep `hprqp_polish_worker` and its `lib/` directory (when present) with the
executable when copying a build.
`make install` installs the worker under `libexec/hprqp`.
Eigen, SuperLU, pocketfft and the JSON parser sources are vendored with their
licenses. The retained Python worker and `polishing/requirements.txt` serve
reference tests only; they are not needed for a normal solver run.

After repairing an installation, use a new dataset output directory: the dataset
runner resumes by skipping already recorded cases, including old timeouts. For a
one-case diagnostic, `HPRQP_DIAGNOSTICS=1 ./build-cuda/solve_qp co5_1.mps` should
show Gram setup succeeding rather than `cudss_not_built`. Ordinary run logs remain
quiet about these extra features unless diagnostics are requested.

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
