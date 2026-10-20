# Experimental GPU SOC support

The SOC prerequisite, ported from `codex/soc-integration`, extends the C/CUDA solver with standard Lorentz
cones, using Julia HPR-SOCP `0cccff3` as the mathematical reference. It solves

```
min  0.5*x'*Q*x + c'*x + constant
s.t. AL <= A_linear*x <= AU
     A_soc*x - rhs_soc in product(SOC)
     l <= x_box <= u
     x_soc in product(SOC)
```

Here Q is symmetric positive semidefinite; zero, diagonal and sparse Q are
supported. SOC(t,v) means `t >= norm(v,2)`, with t first in each block.
The existing C restart, sigma rules, tolerance and check frequency remain the
baseline. This is a generic correctness implementation, not a performance-tuned
port of Julia's specialized cone kernels or controller.

## C API

Include `hprqp.h`, initialize the model using `model_init`, and call
`build_from_QAbc_soc` with the same matrix/vector arguments as `build_from_QAbc`,
followed by a `const HPRQPSOCSpec *` before the output model.

```c
HPRQPSOCSpec cones = {
    .num_var_cones = 1,
    .var_offsets = variable_offsets,
    .num_row_cones = 1,
    .row_offsets = row_offsets,
    .rhs = rhs
};
status = build_from_QAbc_soc(&Q, &A, c, AL, AU, l, u,
                            objective_constant, &cones, &model);
```

- Box variables come first; cone variables occupy a contiguous suffix. Each
  cone's offset array is zero-based, strictly increasing, has `count+1` entries,
  and ends at n for variables or m for rows. Blocks may have any positive
  dimension. For example `{2,5,8}` defines two 3-dimensional cones after two
  box variables. The coordinates' order is preserved in results.
- Linear interval rows come first; affine cone rows occupy a contiguous suffix.
  Their `AL/AU` entries must be `-INFINITY/+INFINITY`. Cone variables similarly
  require unbounded `l/u`; additional bounds can be expressed as linear rows.
  The builder rejects ambiguous box/cone intersections.
- `rhs` contains m finite values with zeros on linear rows. The sign convention
  is **A*x - rhs in SOC**, so affine constants in `A*x + d` use `rhs=-d`.
  It can be NULL when there are no row cones.
- The builder copies all inputs. Use `model_free` to release owned data;
  `model_copy` includes cone metadata and offsets. Cone models retain all rows,
  including zero rows that encode constant cone components.
- A NULL or zero-initialized cone spec uses the existing QP preprocessing and
  GPU path. Existing `build_from_QAbc`, MPS and binding functions retain their
  QP behavior. Rebuild C clients: the exposed `HPRQPModel` layout has new fields.
- Use `params.use_gpu=1`. Requesting the CPU solver for a cone model returns
  `HPRQP_STATUS_NOT_IMPLEMENTED`. The initial SOC surface is the C array API;
  CBF parsing and JuMP/Python cone APIs are not added here.
  `build_from_QAbc_qc_factors` and `build_from_QAbc_qc_diagonal` provide sparse
  low-rank QCQP-to-standard-SOC bridges; see `include/shared/qcqp.h`.

See [solve_socp.c](../examples/solve_socp.c) for a complete example:
`min t`, `(t-2,3,4) in SOC`, whose optimum is t=7.

```sh
make gpu-build
# Select the library directory matching the CUDA toolkit used by make.
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 ./build-cuda/solve_socp
make gpu-test CUDA_ROOT=/usr/local/cuda-12.8
```

CMake with `-DHPRQP_ENABLE_CUDA=ON` also builds `solve_socp` and `test_gpu_soc`.

## Correctness boundaries

The cone path uses the ordinary two-w-update Q-HPR map. It does not use the
mostly-unbounded-variable `noC` shortcut. The interval-only SpMV microbenchmarks
are bypassed for cone models; AUTO selects cuSPARSE. Explicit custom modes use
existing product routines where available, without a claim of tuned throughput.
All check-state outputs are written on every cone iteration.

Each Ruiz/Pock pass replaces scaling factors within a cone by their positive
block maximum. It applies the row factor to the affine offset too. This preserves
standard cones while retaining the C matrix-scaling stages; Julia's aggregation
heuristics and independent generic scaling changes are deferred.

Residual evaluation keeps the iteration slack paired with its multiplier and
uses separate scratch for distance-to-cone feasibility. The dual expression
includes `rhs'*y`, and results/residuals are recovered in original units. The
zero-cone residual path retains upstream behavior. A constant infeasible cone
cannot be silently dropped or reported optimal; this solver still has no new
infeasibility-certificate status.

Tests cover projection boundaries, one-dimensional and long cones, large/small
magnitudes, metadata validation/copying, single-step host parity, scaling round
trips, affine offsets, mixed cones/box/linear constraints, Q=0/diagonal/sparse,
constant rows, and zero-cone API parity. Full solves receive independent
original-unit primal/dual-cone/stationarity/support-gap checks.
