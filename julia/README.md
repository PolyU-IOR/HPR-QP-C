# Julia Wrapper API

## Setup

```sh
cd julia
make setup
```

Test:

```sh
make test
```

Direct demo commands:

```sh
julia --project=. examples/qabc_demo.jl
julia --project=. examples/jump_demo.jl
julia --project=. examples/solve_mps_demo.jl
```

Equivalent Makefile shortcuts:

```sh
make demo-qabc
make demo-jump
make demo-mps
```

## Public API

### Types

- `Model`: opaque model handle
- `Parameters`: solver parameter handle
- `Result`: solve result with copied Julia-owned arrays
- `Optimizer`: JuMP / MathOptInterface optimizer bridge

### Model Construction

- `read_from_mps(path::AbstractString) -> Model`
- `build_from_mps(path::AbstractString) -> Model`
  Alias of `read_from_mps`
- `build_from_QAbc(Q, c, A, AL, AU, l, u; obj_constant=0.0) -> Model`
  Builds a model from sparse or dense `Q` and `A` plus vectors `c`, `AL`, `AU`, `l`, and `u`

### Parameters

- `Parameters(; kwargs...)`

| Keyword | Setter | Values |
| --- | --- | --- |
| `stoptol` | `set_stoptol!(params, value)` | `Real` |
| `max_iter` | `set_max_iter!(params, value)` | `Integer` |
| `use_gpu` | `set_use_gpu!(params, value)` | `Bool` |
| `device_number` | `set_device_number!(params, value)` | `Integer` |
| `time_limit` | `set_time_limit!(params, value)` | `Real` |
| `check_iter` | `set_check_iter!(params, value)` | `Integer` |
| `print_frequency` | `set_print_frequency!(params, value)` | `Integer` |
| `use_scaling` | `set_use_scaling!(params, value)` | `Bool` |
| `verbose` | `set_verbose!(params, value)` | `Bool` |
| `spmv_mode_q` | `set_spmv_mode_q!(params, value)` | `:auto`, `:cusparse`, `:customized` |
| `spmv_mode_a` | `set_spmv_mode_a!(params, value)` | `:auto`, `:cusparse`, `:customized` |

### Solve

- `optimize(model::Model, params::Parameters=Parameters()) -> Result`

| Field | Meaning |
| --- | --- |
| `status_code` | Integer solver status code |
| `status` | Symbolic solver status |
| `iter` | Final iteration count |
| `iter_4` | Iteration when residual first reached `1e-4` |
| `iter_6` | Iteration when residual first reached `1e-6` |
| `time` | Total solve time in seconds |
| `time_4` | Time when residual first reached `1e-4` |
| `time_6` | Time when residual first reached `1e-6` |
| `objective` | Final primal objective |
| `residual` | Final residual metric |
| `gap` | Final relative gap |
| `x` | Primal variable vector |
| `y` | Constraint dual vector |
| `z` | Lower/upper bound dual helper vector |
| `w` | Auxiliary dual vector |