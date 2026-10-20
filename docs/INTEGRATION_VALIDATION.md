# Clean production snapshot qualification — 20 October 2026

## QCQP MPS repair qualification

The original clean package had a reader defect: `QCMATRIX` sections were
silently skipped by the ordinary executable. Historical QCQP benchmarks used a
separate canonical-input bridge and did not test that path.

The repaired `build_from_mps` automatically converts supported diagonal convex
quadratic rows through the existing sparse SOC builder. All **42 raw MPS
models (21 QP and 21 QCQP)** match the saved canonical models: Q/A/AT, objective,
variable/row bounds, cone offsets and RHS, with coefficient tolerance 1e-13.
This is model-equivalence validation, not a fresh full performance benchmark.

The host Make regression suite and all20 CMake tests pass. Focused tests cover
coefficient conventions, duplicate aggregation, inequality reversal, zero
quadratic parts, full QMATRIX objectives, unsupported/nonconvex constraints
and malformed records. Address/undefined-behavior checks pass; local
LeakSanitizer is unavailable under the sandbox's ptrace environment.

On server113 H100 / CUDA13.3, eight checks pass, including:

- default executable and dataset entrypoints on an analytic QCQP;
- QPLIB_2482:361 cones, native OPTIMAL, error9.9547433763e-9;
- invalidation of a legacy false-success CSV and correct resumption thereafter;
- explicit rejection of unsupported input and an ordinary-QP regression;
- CUDA memcheck with zero errors and zero leaked bytes.

The first GPU preflight stopped before running anything because the selected
device became occupied. Validation then used another verified idle device.
The solver, common CPU corrections, Gram rules and GPU kernels are unchanged.
See [input scope](SOC.md#qcqp-mps-input). General non-diagonal QCMATRIX
factorization remains outside the MPS loader; the sparse-factor C API remains
available.

## Earlier clean snapshot qualification

The release is based directly on verified GitHub `main` commit
`3e74c1eaee511879f76a6ebfe0e8070a915b9b65`. The saved production reference is
`a2f2183c5279ae620ee8c416215d221929cbf06c`. Its full42 validation achieved36 native
successes,7 CPU terminations and process SGM10=42.9638940589s. Those are saved
performance results, not a new benchmark of packaging changes.

This snapshot removes the unused per-family polishing drivers and experimental
Julia policy switches. The17 retained CPU numerical/transport/checker files are
byte-identical to the saved version. Two adapter files only lose an unused
dispatcher or change an import; the moved banded factor class is AST-identical.
The active common correction, native certificate, Gram metric, Halpern and
SpMVOp/packed kernels retain their validated rules.

Release checks:

- Make host suite and correction/checker/entrypoint tests pass; CMake host19/19.
- CUDA12.8 compiles the fallback executable without cuDSS. CUDA13.3/cuDSS builds
  with Make and CMake; plain CMake detects CUDA without an enable option.
- 24 GPU qualification commands include17 live contracts and four old/new
  fixed-update byte-parity comparisons (ordinary QP and SOC, Gram off/on).
- Packed-memory checking reports zero errors and zero leaked bytes.
- Default executable and dataset invocations pass with no feature flags. The
  dataset chooses the CUDA binary and records process time and actual features.
- A relocated default native-model driver activates Gram, launches the packaged
  CPU worker without a path override and reaches native1e-8. HPR wins this small
  race; no new live CPU-acceptance claim is inferred from it.
- Four saved-state workers (spatial63, fresh749, late10034, band8803) publish
  independently checked native1e-8 certificates after cleanup. They exercise
  sparse, spatial and banded access; no full original HPR solve is repeated.
- Installed CPU and GPU executables pass a plain model invocation. Server113
  requires its installed CUDA13 compatibility driver library path; a diagnostic
  test without it fails before iteration. A compatible driver is a prerequisite,
  not a feature-enable option.

The dataset resume test recognizes OPTIMAL; failed rows cannot become solved
merely by taking little time. Whole-process and native clocks remain separate.
Python and cuDSS remain installed dependencies, with explicit fallback behavior
as documented in [POLISHING_AND_GRAM.md](POLISHING_AND_GRAM.md).

The following sections are historical integration records. Removed portfolio
paths describe earlier versions only and are not available in this release.

## Historical qualification — 17 October 2026

Base: GitHub production `3e74c1e`. This is an integration qualification, not a
new whole-corpus performance claim. Historical successful CPU states were reused;
no fresh original-instance HPR baselines were run.

- Make CPU suite: all 13 executables passed.
- CMake CPU suite: 15/15 passed, including progress and publication/process tests.
- Independent Python checker: four tests passed, including corrupted primal/dual
  candidates, a false paired gap, nonfinite data and invalid dual support.
- CUDA 12.8/cuDSS 0.7.1 on one H100: all 10 existing GPU test executables passed.
- Seventeen small live integration runs passed analytic objectives, Gram
  activation/fallback, interleaved affine QP rows, and dense update/metric checks
  (maximum normalized defect below `3e-16`). A spatial63 run terminated with an
  independent CPU certificate below `5.4e-10`; returned host vectors were exactly
  the published CPU vectors. These small runs are contract tests, not benchmarks.
- Worker crash and HPR-first cancellation preserved the fixed-length GPU output
  bytes. Ownership tests cover deadline/cancel/reap, wrong token, tampered payload,
  inconsistent tolerance and nonfinite/inaccurate certificates.
- A separate CUDA CMake build without cuDSS passed ordinary-HPR fallback on an
  affine QP. The command-line feature options passed a GPU smoke test.
- Structural AUTO inspection of the six saved panel models selected co5_1 and
  cont5, and declined 10034, 10038, 8803 and QP8585, without using their names.

The packaged simplified worker retained all six saved positive states under an
independent native `1e-8` check:

| Saved instance | Native error after packaged correction |
| --- | ---: |
| 10034 | 1.092e-11 |
| 10038 | 5.685e-14 |
| 8803 | 1.199e-10 |
| QP8585 | 8.253e-9 |
| co5_1 | 1.172e-12 |
| 3029 | 7.485e-10 |

CPU replay time is not an end-to-end GPU speedup. Live behavior on arbitrary
models, alternative libraries, or larger resource budgets remains empirical.
The conservative AUTO rule is a structural generalization of measured evidence;
it is not a universal performance theorem. Historical S707 coverage was 36/42
versus 34/42 saved ordinary results; this source integration has not rerun that
corpus and does not replace its timing table.

Reproduce the self-contained small tests using `POLISHING_AND_GRAM.md`.
Structural numerical-function origins and hashes are in
`POLISHING_SOURCE_PROVENANCE.json`. The research workspace's S718 notebook keeps
the large saved-state replay inputs, complete logs, initial packaging/build
repairs, source hashes and independent audit; those data are deliberately not
embedded in this source commit.

## Default-policy follow-up

The branch is now `test/simple-polishing`. CPU polishing is enabled by default,
and Gram defaults to conservative `AUTO`. This selects the previously qualified
on/auto configuration; it does not broaden the structural gate or change the
numerical correction. Explicit off options remain available. The CPU regression
suite was rebuilt and rerun after changing the defaults; the GPU evidence above
belongs to the preceding integration qualification, not a new benchmark run.

## Equality-QP replacement (S721)

The equality/box backend now uses the exact live-tested S721 fixed-factor
projected correction. It replaces the active-bound completion route; the unused
active-set modules are removed. Other backends, the GPU solver, Gram policy,
progress controller, worker limits and independent terminal checker are unchanged.
The numerical module is byte-identical to the qualified live implementation.

| Instance | New complete process seconds | Previous polisher seconds | Native error |
| --- | ---: | ---: | ---: |
| 10034 | 14.430748 | 10.326141 | 1.0914e-11 |
| 10038 | 14.430358 | 6.671712 | 8.7347e-9 |

Both live first episodes finish within the existing 20-second allowance, factor
once, and return exactly the independently CPU-certified vectors. Three live
surrounding contracts pass. Six small local contracts and two saved original
candidate replays also pass. The branch's polishing test target includes the
independent checker, process/progress tests and fixed-factor known-solution,
dispatch, unchanged-input, singular-factor, zero-trace and expiry checks.

This is a successful simplification with higher cost than the previous
specialized polisher, not a new speedup over it. Full process time includes all
overhead and counts concurrent CPU/GPU work once. Saved comparisons use the same
GPU executable/device but different sessions.

Before replacement, S719 tested the 36 previously successful cases: 36/36
solved, six CPU polishing terminations (10034, 10038, QP8585, 3029, co5_1, 8803),
and complete-process SGM10 23.516378 seconds. Substituting the two measured new
times gives 24.047539 seconds. Charging the other six corpus cases 1000 seconds
each gives a 42-case composite SGM10 of **45.259265 seconds**, versus 44.519514
before substitution. The composite has 36/42 successes and six polishing
terminations. It is a saved-result calculation, not a fresh full-corpus run of
this commit; other eligible equality-QP paths have not been rerun after replacing
their backend. SGM10 uses shift 10, and successful process times are not clipped.
Ordinary QP references have native time only, so no ordinary full-process
combined baseline is claimed.

## S783 common implementation integration (2026-10-19)

This supersedes the earlier portfolio dispatcher. All 23 CPU files and the
continued equation-progress header match the tested S781/S782 candidate. The
GPU iteration/Halpern, control and SpMVOp sources also match. Production
packaging defaults the cumulative worker budget to zero and automatically
attempts the validated spatial Gram inverse within unchanged Gram eligibility;
its diagnostic output is optional. No new correction method or threshold.

Qualification on server 113 H100:

- CUDA 12.8 without cuDSS and CUDA 13.3 with cuDSS: six GPU tests each pass;
  focused SpMVOp memcheck reports zero errors/leaks.
- Fourteen native model runs pass, including affine/mixed/dependent-row and
  Gram activation/decline contracts. One old timing assertion was repaired:
  spatial63 is solved by HPR in 0.84 seconds before the worker finishes, which
  is valid. Native thresholds and accepted-payload checks were not relaxed.
- Worker crash and cancellation preserve exact fixed-length GPU vectors.
- Default spatial access and an unwritable optional report produce exactly the
  saved S781 source's 3000-update vectors. Explicit direct-Gram opt-out passes.
- Four real common-worker publications (sparse, spatial and banded access)
  reproduce saved candidate bytes; independent native certificate audits pass.
- CPU progress, publication, checker, historical helper and common-correction
  contracts pass. Five analytic QPs check both derivative actions and native
  optimality. A local missing-NumPy environment and a continuation missing CUDA
  environment were corrected without solver changes; failed evidence remains.

Evidence: workspace `Oct19_hprqp_s783_common_integration/ROOT_FINAL_AUDIT.json`
and preserved build/qualification logs. This is integration qualification,
not a new benchmark speed measurement or fresh 42-instance score. The latest
mixed-run 42-case estimate remains 36 solved, SGM10 43.480505 seconds.
