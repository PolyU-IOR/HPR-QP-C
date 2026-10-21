# HPR-QP-C

A CUDA/C implementation of the Halpern Peaceman--Rachford (HPR) method for solving quadratic programming (QP) problems on GPUs.

## Quick start

From the project directory, with a compatible CUDA toolkit and NVIDIA driver:

```sh
make && \
./build-cuda/solve_qp model.mps --use-gpu 1 --device-number 0 --stoptol 1e-8 --time-limit 1000
```

Replace `model.mps` with your input file and `0` with the desired GPU index.
The command explicitly launches the GPU executable. Native CPU polishing,
automatic Gram selection and supported GPU optimizations are enabled by default.
See [Install](#install) below for prerequisites and dependency setup.

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
BLAS/LAPACK prerequisites into this checkout automatically. The default
`make` build uses the C/C++ compilers directly and does not require CMake.
With a compatible CUDA toolkit and driver installed:

```sh
make
```

For native CPU polishing, `make` first reuses working BLAS/LAPACK libraries,
including an existing private installation under `.deps/native/`.
On Debian/Ubuntu Linux x86_64, missing libraries and their Fortran/GCC runtimes
are downloaded from the distribution's APT metadata with verified hashes and
installed privately under `.deps/native/`. This requires `apt-get`/`dpkg-deb`,
usable package indexes and network access. For standard Ubuntu archive URLs,
the installer tries official HTTPS mirror alternatives while preserving the
exact APT package path, size and checksum. The installation receipt records both
the metadata URL and the successful download URL. It does not run `sudo`, execute
package installation scripts or change system packages.
`make setup-native` prepares missing native libraries explicitly.
`AUTO_INSTALL_NATIVE_DEPS=0` disables native dependency downloads; existing
usable system or private libraries are still reused. Other platforms must
provide compatible BLAS/LAPACK libraries themselves. No platform-specific binary
runtime is bundled in the source tree. CMake remains an optional alternative
build system; plain `make` does not invoke or install it.

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

Downloads verify HTTPS certificates and archive checksums. The installers use
the standard system CA bundle even when a custom OpenSSL installation has an
empty default trust store. Explicit `SSL_CERT_FILE` / `SSL_CERT_DIR` settings
remain authoritative. If a managed network needs its own CA, ask its
administrator for the trusted bundle and set `SSL_CERT_FILE` before `make`;
do not disable certificate verification.

Linux GPU builds embed the selected CUDA/cuDSS library paths with priority over
conflicting `LD_LIBRARY_PATH` entries. This prevents an older cuSPARSE with the
same library name from replacing the version used to build SpMVOp support.
After updating the source, rerun `make`; its build configuration automatically
invalidates the old link policy. Only launch the solver after `make` succeeds
(the quick-start `&&` enforces this). A failed build may leave an old executable.

If several CUDA toolkits are installed, use the same `CUDA_ROOT` in both commands:

```sh
make CUDA_ROOT=/usr/local/cuda-13.3
make check-cudss CUDA_ROOT=/usr/local/cuda-13.3
```

Before building GPU targets, `make` tests CUDA startup without shell library
overrides. It prefers the system driver. If that fails and the selected toolkit
has an existing `compat/libcuda.so.1`, it tests that library in a separate process
and links it explicitly only after successful initialization, so vendor-library
startup cannot load a different driver first. The direct executable,
shared library and Python dataset script then use the same verified runtime;
no compatibility `LD_LIBRARY_PATH` export is needed. The build receipt is
`.deps/cuda-runtime.json`. CMake performs the same check at configuration.

If neither driver works, the build stops with the actual CUDA error. Choose a
driver-compatible toolkit (for example `make CUDA_ROOT=/usr/local/cuda-12.8`)
or ask the administrator for a supported driver/compatibility package. No build
can fix an unsupported driver without a working alternative; `make` never
installs or changes drivers. Re-run `make` after moving a build to another host
or changing CUDA/driver installations. Previously failed builds may leave old
executables, so use the quick-start `make && ...` command.

When CUDA reports no visible GPU, compilation is allowed with an explicit
unverified-startup warning. `make CUDA_DRIVER_CHECK=0` (CMake:
`-DHPRQP_CHECK_CUDA_DRIVER=OFF`) also permits an intentional compile-only build;
it is not a fix for runtime incompatibility. Validate on the GPU host before use.

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
