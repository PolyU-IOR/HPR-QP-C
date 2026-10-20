# Native worker and cuDSS build validation

Validated on 2026-10-20. This is a deployment and numerical-port qualification,
not a new 42-instance performance benchmark.

## Dependency delivery

An actual `make` invocation with CUDA 12.8 and no matching local cuDSS downloaded
NVIDIA cuDSS 0.7.1.4, verified its published SHA-256, reparsed the build and
compiled Gram support. CUDA 13.0 discovery and Gram compilation also passed.
The CUDA 13 check was a build check, not a GPU runtime test.

Tests cover CUDA-major matching, a wheel containing only `libcudss.so.0`, a
single installation for parallel Make targets, rebuilds after capability
changes, checksum rejection, safe archive paths and preservation of an existing
installation. Default GPU builds fail clearly if required cuDSS is unavailable
and automatic installation is disabled.

## Common correction and independent checking

The native worker was tested against the retained Python implementation as an
independent oracle. Tests cover general quadratic objectives, affine and mixed
rows, dependent equations, boxes, all SOC derivative regions, compatible banded
solves and spatial elimination on two unused small grids. Corrupt vectors,
nonfinite values, invalid dual support, wrong request identity and malformed
transport cannot publish a certificate.

The final native executable also certifies all seven previously accepted saved
HPR points: 10034, 10038, QP8585, 3029, co5_1, 8803 and 2626. Each serialized
candidate independently passes the retained native-model CPU checker.
These are saved-point retention results, not complete-solve timings.

## Live GPU and installation

Local CUDA 12.8 / RTX 3080 tests use a solver environment with
`PATH=/nonexistent` and no `PYTHONPATH`. The common native worker runs
successfully without an interpreter.

| Live test | Result |
| --- | --- |
| Small public MPS QP and QCQP | Native optimum; quiet default logs |
| Small mixed-QP and spatial Gram models | Gram active; native optimum |
| Original co5_1 through native-model driver | Gram active; CPU-certified termination; error `2.30e-10`; complete process `94.745 s` |
| Original QP8585 through public dataset script | `polished_optimal`; error `4.05e-9`; solver subprocess `14.472 s` |
| Dataset resume | No duplicate solve |
| Failed native worker at fixed update count | Returned GPU vectors byte-identical to polishing-disabled control |
| Relocated installation | Native worker discovered without source fallback; valid CPU publication; installed CLI passes |

The complete process clocks include setup, all failed worker attempts,
publication and cleanup. Overlapping work is counted once. The co5_1 harness
initially supplied the wrong binary transport and was rejected before any
numerical work; the corrected continuation preserved that failure and reused
the four completed small checks. GPU timings here are not comparable to the
earlier H100 benchmark timings.

## Reproducing source-level checks

```sh
make
python3 tests/test_cudss_discovery.py
python3 tests/test_cudss_setup.py
# NumPy/SciPy are needed for these optional independent test oracles only:
python3 -m pip install -r polishing/requirements.txt
make test-native-polishing
make test-polishing
```

Default solver execution uses the compiled C++ worker. The Python source remains
an explicit test/reference option. Changing BLAS, LAPACK or factorization
versions can change rounding and trajectories; the port preserves the common
equations and acceptance rules rather than promising byte-identical successful
polishing trajectories across libraries.

## Automatic native prerequisite delivery

The follow-up installation repair was tested on Ubuntu 22.04 server dantzig
without system CMake or BLAS/LAPACK. Plain `make` installed checksum-verified
CMake 3.28.4 and the distribution's native runtime libraries locally, then
built the solver and worker. All numerical source files remained unchanged.
Seven installer contract tests cover reuse, offline behavior, checksum and
archive rejection, and distinguishing missing libraries from unrelated errors.

The worker resolves every private BLAS/LAPACK/Fortran dependency after copying
it and its `lib/` directory to a new location, without a native library
environment override. A second build with both automatic downloads disabled
succeeds and leaves the worker binary unchanged. Public dataset small QP/QCQP
checks pass at native `1e-8`. These are installation checks, not a new corpus
performance result.

On this host, CUDA 13.3 requires its already-installed compatibility driver
path; without it, both initial smoke checks returned before any iteration.
Those initialization failures are preserved separately from the successful
checks using the compatible runtime. No driver or system package was modified.
