# Simplified CPU polishing and equality Gram scaling

This experimental branch starts from GitHub production commit `3e74c1e` and
packages two independently selectable features. It also includes their native
SOC and low-rank QCQP-to-SOC prerequisites. `params_set_defaults` enables CPU
polishing and selects `HPRQP_GRAM_AUTO`; the structural eligibility and fallback
guards still decide whether Gram actually activates. Set `polishing=0` and
`gram_mode=HPRQP_GRAM_OFF` (CLI: `--polishing 0 --gram off`) for ordinary HPR.
These defaults apply to the GPU solver; CPU solves do not launch the worker.
Recompile clients:
`HPRQPModel`, `HPRQPParams` and `HPRQPResults` have extended layouts.

## Build and use

Linux, a 64-bit host, C++17 and CUDA are required for live GPU polishing.
The worker needs Python 3, NumPy and SciPy; the owner needs OpenSSL's
`libcrypto.so.3` or `libcrypto.so.1.1` for payload verification. Python is started
only when an eligible snapshot is available. Gram additionally needs cuDSS.
The qualified environment used CUDA 12.8, cuDSS 0.7.1, NumPy 2.3.4 and SciPy 1.16.2.
Broader ranges in `polishing/requirements.txt` are installation constraints,
not claims of identical numerical trajectories on every version.

```sh
python3 -m pip install -r polishing/requirements.txt
make
./build-cuda/solve_qp model.mps
python3 scripts/run_dataset.py /path/to/mps /path/to/results
# Equivalently: make dataset DATASET_DIR=/path/to/mps RESULT_DIR=/path/to/results
```

No feature-enable arguments are required. The executable and dataset script share
`params_set_defaults`: polishing on, conservative Gram auto, automatic SpMV
selection, and packed/fused sparse updates where eligible. The dataset script
prefers `build-cuda/solve_qp`; `make run` and `make dataset` select the CUDA build
when a toolkit is available. CPU-only builds cannot run GPU features.

Make and CMake discover cuDSS from standard prefixes, the selected toolkit,
`/opt/nvidia/cudss`, or a matching NVIDIA Python wheel. A nonstandard installation
can be located with the build hint `CUDSS_ROOT`; this is a dependency location,
not a runtime enable flag. Without cuDSS, Gram reports a decline and HPR continues.
CMake also enables CUDA automatically when a CUDA compiler is available; explicit
`-DHPRQP_ENABLE_CUDA=OFF` builds CPU only. SpMVOp needs CUDA 13.3 or newer and falls
back to the existing cuSPARSE path on older toolkits or unsupported operations.

The CUDA toolkit must have a compatible driver. On hosts using NVIDIA's forward
compatibility package, configure that package's library path as part of the CUDA
installation. Automatic feature selection does not repair an incompatible driver.

The CPU worker uses `python3` on PATH. Install its NumPy/SciPy requirements in that
interpreter (or activate the intended virtual environment). Source builds locate
the source-tree worker automatically; installed executables and shared libraries
locate `share/hprqp/polishing` relative to their installed location, including after
relocation. A custom `polish_python` or `polish_worker` is an optional override.
Do not deploy the executable alone without its worker package and dependencies.

The C API and existing Python/Julia QP wrappers inherit the same central defaults.
Recompile C clients after this layout change. Native SOC and sparse-factor QCQP
construction remain exposed through the C array API; the dataset runner reads
MPS QPs and does not add a QPLIB QCQP parser.

```c
HPRQPParams params;
params_set_defaults(&params);
HPRQPStatus status = solve(&model, &params, &results);
```

## CPU worker contract

1. The owner observes the existing GPU residual checks. After seven check
   intervals, a crossing of native total error or `max(primal, dual)` through
   decades from `1e-2` can start a worker. Total-error admission stops at
   `10*stoptol`; equation admission continues to
   `min(10*stoptol, 64*DBL_EPSILON)` while native termination still fails. A solved point
   does not start a worker. There is no elapsed-time trigger.
2. One private process receives the immutable **full unscaled native model**
   (`Q,A,c`, bounds, affine offsets, cones and objective constant) and a copied
   unscaled HPR `(x,w,z,y)` point. Native means the solver's lifted conic model
   for a QCQP, not an additional original-QCQP checker.
3. All models use the same unit-scale projected-KKT residual and one frozen
   derivative from the copied HPR point. The worker allows eight full linear
   corrections, checks the same native certificate after each, and rejects
   inaccurate linear actions or growth beyond 100 times the initial residual.
   Private residual and iterate accumulation use extended precision uniformly;
   linear factors remain double precision. Sparse factors, checked banded
   factors, exact elimination and nullspace/preconditioning access change only
   how that same linear system is solved. Structure is detected from matrix
   data, never an instance name. No adapter owns a different residual,
   continuation rule or candidate construction. There is no active-bound
   update, iterative SSN refresh, history acceleration or model-started solver.
   See the [common correction equations](common_correction.md).
4. A separate CPU checker rereads the serialized candidate, recomputes native
   primal feasibility, dual stationarity/domain feasibility and support-based
   relative objective gap, and requires every category to be finite and at
   most `stoptol`. It does not trust the correction's cached slack or certificate.
   Explicit private dual-domain cleanup precedes checking. The GPU owner checks
   the certificate's token, dimensions, tolerance, gap consistency and payload
   SHA-256 before importing the host result.
5. Only a certified terminal candidate can replace the returned host vectors.
   **No candidate, partial progress, penalty or active set is uploaded to GPU.**
   GPU iterates, restart anchors and control state continue unchanged until
   termination. HPR winning the race cancels/reaps the child. Failed, expired
   and unsupported episodes leave HPR running; pending crossings coalesce and
   the next episode uses the latest eligible HPR point.

Resource options are operational, not declarations that a direction is hopeless:

| C parameter | Default | Zero means |
| --- | ---: | --- |
| `polish_max_attempts` | 12 | no attempt-count cap |
| `polish_episode_seconds` | 20 | remaining HPR time budget |
| `polish_total_seconds` | 0 | no cumulative worker-time cap |
| `polish_memory_bytes` | 8 GiB | no separate address-space cap |

The solver's global time and iteration limits remain in effect. Changing these
limits changes the policy and requires separate performance evaluation.
`polish_directory` selects an existing parent directory (default `/tmp`);
`polish_keep_files=1` preserves snapshots, events and rejection reports.
Defaults use a private directory and delete it on completion. Failed dependency
loading disables the worker, with a diagnostic when `verbose=1`.
Variable-cone models currently decline CPU polishing.

## Gram policy and mathematical meaning

`HPRQP_GRAM_OFF` preserves the original row update. `AUTO` recognizes homogeneous
row-SOC spatial models with a verified five-point tensor Laplacian state block,
positive diagonal state curvature and three-dimensional cones. This is a
conservative heuristic motivated by the co5_1/cont5 evidence, not a guarantee
for every matching PDE. Pure QPs are not automatically activated.

`REQUEST` attempts broader supported models: affine, interleaved equality/interval
QP rows and general PSD Q; the legacy conic path requires a homogeneous equality
prefix, diagonal Q and no variable cones. Factorization/inverse validation,
unsupported structure, missing cuDSS or excessive Gram assembly work can decline
before iteration. Setup costs still count. Once active, a linear-solver error
propagates; the algorithm never silently switches row metrics in mid-run.

For `A=[E;C]`, `G=EE'`, and `beta >= ||C||_2^2`, the tested row majorizer is
`diag(2G, 2 beta I)`. It is ordinary scalar-majorized HPR in coordinates whitened
by `diag(L^-1, beta^-1/2 I)`, where `LL'=G`. Its equality update uses a Gram solve;
its rest-row projections and HPR splitting retain their form. Restart and sigma
geometry are transported consistently, and final stopping remains in unscaled
native units. The factor two safeguards coupling between the row blocks: this
is an ADMM-style **proximal** equality update, not an undamped equality ADMM
subproblem. See the [formal derivation](equality_gram_scaling.tex).

The constant factor is reused across sigma changes. Eligible spatial models
also attempt the validated tensor/Schur Gram inverse. It must pass structural
identities and four independent right-hand-side accuracy checks against the
direct factor; otherwise the direct factor is retained. This is equivalent
linear access, not a new row update. `HPRQP_SPATIAL_GRAM=0` disables that access;
`HPRQP_SPATIAL_GRAM_REPORT=/path/report.json` optionally saves diagnostics.
Missing or unwritable diagnostic output does not change numerical selection.
Broad Gram showed
coverage regressions in the research corpus, so theoretical eligibility alone
is not used as the automatic policy.

## Results and testing

`results.polish_attempts`, `polish_accepted` and `gram_active` distinguish the
requested features from actual use. `results.total_time` is the complete GPU
`solve()` call, including selection/setup, export, overlapping worker execution,
publication and cleanup. It excludes the caller's model reading/building and
process startup. `results.time` retains the existing HPR-loop time convention.
Use external process wall time for full application comparisons and do not add
CPU-worker time again when it overlaps GPU work.

```sh
make test-cpu
make test-polishing
make gpu-test
make build-cuda/polish_native_driver
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=polishing python3 tests/polishing/test_gpu.py \
    build-cuda/polish_native_driver /tmp/hprqp-new-qualification
```

Select a device you are authorized to use; the last output directory must not
already exist. Tests cover dense Gram equations, activation/fallback, analytic
QP objectives, exact terminal payload return, byte-identical GPU trajectories
after worker failure, process cancellation, publication tampering and independent
native checking. See [integration evidence](INTEGRATION_VALIDATION.md).

The dataset CSV records `process_time` (launch through exit), `total_time` (the
solve API), and `alg_time` (native loop) separately, plus actual Gram activation
and polishing attempts/acceptance. `OPTIMAL` controls solved counts. SGM10 uses
the configured time limit for failed rows and retains successful process times
even when they exceed that limit. Historical rows lacking a required successful
clock leave that clock's aggregate blank; no timing values are invented. Resume
skips completed OPTIMAL, MAX_ITER and TIME_LIMIT records.
