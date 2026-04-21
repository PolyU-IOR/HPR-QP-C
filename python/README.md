# Python Wrapper API

## Setup

```sh
cd python
make setup
```

Test:

```sh
make test
```

Direct demo commands:

```sh
python examples/build_qabc_demo.py
python examples/solve_mps_demo.py
python examples/cvxpy_demo.py
```

Equivalent Makefile shortcuts:

```sh
make demo-qabc
make demo-mps
make demo-cvxpy
```

## Public API

### Types

- `Model`: opaque model handle
- `Parameters`: solver parameter object
- `Result`: solve result with copied Python-owned lists
- `CVXPYBuildInfo`: metadata returned by `build_from_cvxpy_problem`

### Model Construction

- `read_from_mps(path) -> Model`
- `build_from_mps(path) -> Model`
  Alias of `read_from_mps`
- `build_from_qabc(Q, c, A, AL, AU, l, u, obj_constant=0.0) -> Model`
- `build_from_QAbc(...) -> Model`
  Alias of `build_from_qabc`

`Q` and `A` may be dense row-major matrices or SciPy sparse matrices.

### Parameters

- `Parameters(**kwargs)`

| Keyword | Method | Values |
| --- | --- | --- |
| `stoptol` | `set_stoptol(value)` | `float` |
| `max_iter` | `set_max_iter(value)` | `int` |
| `use_gpu` | `set_use_gpu(value)` | `bool` |
| `device_number` | `set_device_number(value)` | `int` |
| `time_limit` | `set_time_limit(value)` | `float` |
| `check_iter` | `set_check_iter(value)` | `int` |
| `print_frequency` | `set_print_frequency(value)` | `int` |
| `use_scaling` | `set_use_scaling(value)` | `bool` |
| `verbose` | `set_verbose(value)` | `bool` |
| `spmv_mode_q` | `set_spmv_mode_q(value)` | `'auto'`, `'cusparse'`, `'customized'`, or `SPMV_MODE_*` |
| `spmv_mode_a` | `set_spmv_mode_a(value)` | `'auto'`, `'cusparse'`, `'customized'`, or `SPMV_MODE_*` |

Also available:

- `set_defaults()`

### Solve

- `optimize(model, params=None) -> Result`

| Field | Meaning |
| --- | --- |
| `status_code` | Integer solver status code |
| `status` | String solver status |
| `iter` | Final iteration count |
| `iter_4` | Iteration when residual first reached `1e-4` |
| `iter_6` | Iteration when residual first reached `1e-6` |
| `time` | Total solve time in seconds |
| `time_4` | Time when residual first reached `1e-4` |
| `time_6` | Time when residual first reached `1e-6` |
| `objective` | Final primal objective |
| `residual` | Final residual metric |
| `gap` | Final relative gap |
| `x` | Primal variable list |
| `y` | Constraint dual list |
| `z` | Lower/upper bound dual helper list |
| `w` | Auxiliary dual list |

### CVXPY Integration

- `build_from_cvxpy_problem(problem) -> (Model, CVXPYBuildInfo)`
- `solve_cvxpy_problem(problem, params=None, assign_values=True) -> Result`

| Field | Meaning |
| --- | --- |
| `variable_offsets` | Map from CVXPY variable id to flat solver offset |
| `variable_shapes` | Original CVXPY variable shapes |
| `num_variables` | Total flattened variable count |
| `num_constraints` | Total canonical constraint count |
| `objective_constant` | Constant objective offset |