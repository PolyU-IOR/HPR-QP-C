# Clean production scope

Upstream base: `PolyU-IOR/HPR-QP-C`, `main` at
`3e74c1eaee511879f76a6ebfe0e8070a915b9b65`, verified20October2026.
Branch: `test/simple-polishing`. Previous local history remains preserved at
`codex/archive-simple-polishing-a2f2183`; this release is a single local commit
on the upstream base. Nothing is pushed by the packaging procedure.

| Included change | Default behavior |
| --- | --- |
| Common CPU polishing | Enabled; progress admits snapshots, one frozen derivative, common correction/rejection/candidate rules and independent native certificate. Only a terminal certificate replaces returned host vectors; no GPU feedback. |
| Equality Gram scaling | AUTO; conservative matrix-based eligibility and validated direct/spatial inverse, with fallback on unsupported models or unavailable dependency. |
| SpMVOp | Automatic in supported cuSPARSE paths on CUDA13.3+, older-toolkit fallback retained. |
| Packed/fused GPU access | Enabled subject to common shape, padding and memory guards; scaled coefficients refreshed. |
| Native SOC / low-rank QCQP-to-SOC | Required infrastructure for the validated QCQP cases; sparse factors retained. |
| Build, installation, entrypoints, tests and notes | Automatic CUDA/cuDSS discovery, relative installed-worker lookup, default executable/dataset selection and honest cost reporting. |

The snapshot excludes the old polishing portfolio/history acceleration,
Julia-policy ablations, new sigma/history/coupled-metric experiments, derivative
refresh, active-bound updates and independent model-started solvers. The
pre-existing CPU solver remains available; these GPU features do not turn a
CPU-only binary into a GPU solver.

`params_set_defaults` is shared by the C executables and bindings. The dataset
runner does not override polishing, Gram or SpMV defaults. Explicit opt-outs and
resource overrides remain available for controlled experiments; ordinary use
needs no feature-enable arguments. Dependencies still have to be installed.
See [usage](POLISHING_AND_GRAM.md) and [validation](INTEGRATION_VALIDATION.md).
