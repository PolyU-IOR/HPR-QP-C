from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

STATUS_OK = 0
STATUS_INVALID_ARGUMENT = 1
STATUS_ALLOCATION_FAILED = 2
STATUS_NOT_IMPLEMENTED = 3
STATUS_MAX_ITER = 4
STATUS_TIME_LIMIT = 5
STATUS_OPTIMAL = 6
STATUS_CONTINUE = 7

SPMV_MODE_AUTO = 0
SPMV_MODE_CUSPARSE = 1
SPMV_MODE_CUSTOMIZED = 2


@dataclass(frozen=True)
class Result:
    status_code: int
    status: str
    iter: int
    iter_4: int
    iter_6: int
    time: float
    time_4: float
    time_6: float
    objective: float
    residual: float
    gap: float
    x: list[float]
    y: list[float]
    z: list[float]
    w: list[float]


@dataclass(frozen=True)
class CVXPYBuildInfo:
    variable_offsets: dict[int, int]
    variable_shapes: dict[int, tuple[int, ...]]
    num_variables: int
    num_constraints: int
    objective_constant: float


class _Library:
    _instance: ctypes.CDLL | None = None
    _path: str | None = None

    @classmethod
    def root(cls) -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def candidates(cls) -> list[Path]:
        root = cls.root()
        if os.name == "nt":
            names = ["hprqp.dll", "libhprqp.dll"]
        elif sys.platform == "darwin":
            names = ["libhprqp.dylib"]
        else:
            names = ["libhprqp.so"]
        candidates: list[Path] = []
        env_path = os.environ.get("HPRQP_LIBRARY_PATH")
        if env_path:
            candidates.append(Path(env_path))
        for directory in [
            root / "build-cuda-shared",
            root / "build-shared",
            root / "build-cuda",
            root / "build",
            root / ".install-gpu-test" / "lib",
            root / ".install-cpu-test" / "lib",
        ]:
            for name in names:
                candidates.append(directory / name)
        return candidates

    @classmethod
    def load(cls) -> ctypes.CDLL:
        last_error: Exception | None = None

        if cls._instance is not None:
            return cls._instance

        for candidate in cls.candidates():
            if candidate.is_file():
                try:
                    lib = ctypes.CDLL(str(candidate))
                    cls._configure(lib)
                except (AttributeError, OSError) as exc:
                    last_error = exc
                    continue

                cls._instance = lib
                cls._path = str(candidate)
                return cls._instance

        fallback = ctypes.util.find_library("hprqp")
        if fallback:
            try:
                lib = ctypes.CDLL(fallback)
                cls._configure(lib)
            except (AttributeError, OSError) as exc:
                last_error = exc
            else:
                cls._instance = lib
                cls._path = fallback
                return cls._instance

        message = (
            "Unable to locate libhprqp. Build it first with `make gpu-build-shared` or `make build-shared`, "
            "or set HPRQP_LIBRARY_PATH to the shared library path."
        )
        if last_error is not None:
            message = f"{message} Last load error: {last_error}"
        raise RuntimeError(message)

    @classmethod
    def _configure(cls, lib: ctypes.CDLL) -> None:
        void_pp = ctypes.POINTER(ctypes.c_void_p)
        size_p = ctypes.POINTER(ctypes.c_size_t)
        double_p = ctypes.POINTER(ctypes.c_double)

        lib.hprqp_status_name.argtypes = [ctypes.c_int]
        lib.hprqp_status_name.restype = ctypes.c_char_p
        lib.hprqp_gpu_backend_available.argtypes = []
        lib.hprqp_gpu_backend_available.restype = ctypes.c_int

        lib.hprqp_model_create.argtypes = [void_pp]
        lib.hprqp_model_create.restype = ctypes.c_int
        lib.hprqp_model_destroy.argtypes = [ctypes.c_void_p]
        lib.hprqp_model_destroy.restype = None
        lib.hprqp_model_read_from_mps.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.hprqp_model_read_from_mps.restype = ctypes.c_int
        lib.hprqp_model_build_from_qabc_csr.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, size_p, size_p, double_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, size_p, size_p, double_p,
            double_p, double_p, double_p, double_p, double_p, ctypes.c_double,
        ]
        lib.hprqp_model_build_from_qabc_csr.restype = ctypes.c_int
        lib.hprqp_model_num_variables.argtypes = [ctypes.c_void_p]
        lib.hprqp_model_num_variables.restype = ctypes.c_size_t
        lib.hprqp_model_num_constraints.argtypes = [ctypes.c_void_p]
        lib.hprqp_model_num_constraints.restype = ctypes.c_size_t

        lib.hprqp_params_create.argtypes = [void_pp]
        lib.hprqp_params_create.restype = ctypes.c_int
        lib.hprqp_params_destroy.argtypes = [ctypes.c_void_p]
        lib.hprqp_params_destroy.restype = None

        for name in [
            ("hprqp_params_set_defaults", [ctypes.c_void_p]),
            ("hprqp_params_set_stoptol", [ctypes.c_void_p, ctypes.c_double]),
            ("hprqp_params_set_max_iter", [ctypes.c_void_p, ctypes.c_size_t]),
            ("hprqp_params_set_use_gpu", [ctypes.c_void_p, ctypes.c_int]),
            ("hprqp_params_set_device_number", [ctypes.c_void_p, ctypes.c_int]),
            ("hprqp_params_set_time_limit", [ctypes.c_void_p, ctypes.c_double]),
            ("hprqp_params_set_check_iter", [ctypes.c_void_p, ctypes.c_size_t]),
            ("hprqp_params_set_print_frequency", [ctypes.c_void_p, ctypes.c_int]),
            ("hprqp_params_set_use_scaling", [ctypes.c_void_p, ctypes.c_int]),
            ("hprqp_params_set_verbose", [ctypes.c_void_p, ctypes.c_int]),
            ("hprqp_params_set_spmv_mode_q", [ctypes.c_void_p, ctypes.c_int]),
            ("hprqp_params_set_spmv_mode_a", [ctypes.c_void_p, ctypes.c_int]),
        ]:
            func = getattr(lib, name[0])
            func.argtypes = name[1]
            func.restype = ctypes.c_int

        lib.hprqp_results_create.argtypes = [void_pp]
        lib.hprqp_results_create.restype = ctypes.c_int
        lib.hprqp_results_destroy.argtypes = [ctypes.c_void_p]
        lib.hprqp_results_destroy.restype = None
        lib.hprqp_optimize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.hprqp_optimize.restype = ctypes.c_int

        for name, restype in [
            ("hprqp_results_status", ctypes.c_int),
            ("hprqp_results_num_variables", ctypes.c_size_t),
            ("hprqp_results_num_constraints", ctypes.c_size_t),
            ("hprqp_results_iter", ctypes.c_size_t),
            ("hprqp_results_iter_4", ctypes.c_size_t),
            ("hprqp_results_iter_6", ctypes.c_size_t),
            ("hprqp_results_time", ctypes.c_double),
            ("hprqp_results_time_4", ctypes.c_double),
            ("hprqp_results_time_6", ctypes.c_double),
            ("hprqp_results_objective", ctypes.c_double),
            ("hprqp_results_residual", ctypes.c_double),
            ("hprqp_results_gap", ctypes.c_double),
        ]:
            func = getattr(lib, name)
            func.argtypes = [ctypes.c_void_p]
            func.restype = restype

        for name in [
            "hprqp_results_x_data",
            "hprqp_results_y_data",
            "hprqp_results_z_data",
            "hprqp_results_w_data",
        ]:
            func = getattr(lib, name)
            func.argtypes = [ctypes.c_void_p]
            func.restype = ctypes.POINTER(ctypes.c_double)


def _lib() -> ctypes.CDLL:
    return _Library.load()


def status_name(status: int) -> str:
    return _lib().hprqp_status_name(int(status)).decode("utf-8")


def gpu_backend_available() -> bool:
    return bool(_lib().hprqp_gpu_backend_available())


def _raise_for_status(code: int, context: str) -> int:
    if code != STATUS_OK:
        raise RuntimeError(f"{context} failed with status {status_name(code)}")
    return code


def _bool_to_int(value: bool) -> int:
    return 1 if value else 0


def _spmv_mode_code(value: int | str) -> int:
    if value in (SPMV_MODE_AUTO, "auto"):
        return SPMV_MODE_AUTO
    if value in (SPMV_MODE_CUSPARSE, "cusparse"):
        return SPMV_MODE_CUSPARSE
    if value in (SPMV_MODE_CUSTOMIZED, "customized"):
        return SPMV_MODE_CUSTOMIZED
    raise ValueError("Unsupported SpMV mode. Expected 'auto', 'cusparse', or 'customized'.")


class Model:
    def __init__(self) -> None:
        handle = ctypes.c_void_p()
        _raise_for_status(_lib().hprqp_model_create(ctypes.byref(handle)), "hprqp_model_create")
        self._handle = handle

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            _lib().hprqp_model_destroy(handle)
            handle.value = None

    @property
    def num_variables(self) -> int:
        return int(_lib().hprqp_model_num_variables(self._handle))

    @property
    def num_constraints(self) -> int:
        return int(_lib().hprqp_model_num_constraints(self._handle))


class Parameters:
    def __init__(self, **kwargs: object) -> None:
        handle = ctypes.c_void_p()
        _raise_for_status(_lib().hprqp_params_create(ctypes.byref(handle)), "hprqp_params_create")
        self._handle = handle
        for name, value in kwargs.items():
            getattr(self, f"set_{name}")(value)

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            _lib().hprqp_params_destroy(handle)
            handle.value = None

    def _call(self, func_name: str, *args: object) -> "Parameters":
        _raise_for_status(getattr(_lib(), func_name)(self._handle, *args), func_name)
        return self

    def set_defaults(self) -> "Parameters":
        return self._call("hprqp_params_set_defaults")

    def set_stoptol(self, value: float) -> "Parameters":
        return self._call("hprqp_params_set_stoptol", float(value))

    def set_max_iter(self, value: int) -> "Parameters":
        return self._call("hprqp_params_set_max_iter", int(value))

    def set_use_gpu(self, value: bool) -> "Parameters":
        return self._call("hprqp_params_set_use_gpu", _bool_to_int(bool(value)))

    def set_device_number(self, value: int) -> "Parameters":
        return self._call("hprqp_params_set_device_number", int(value))

    def set_time_limit(self, value: float) -> "Parameters":
        return self._call("hprqp_params_set_time_limit", float(value))

    def set_check_iter(self, value: int) -> "Parameters":
        return self._call("hprqp_params_set_check_iter", int(value))

    def set_print_frequency(self, value: int) -> "Parameters":
        return self._call("hprqp_params_set_print_frequency", int(value))

    def set_use_scaling(self, value: bool) -> "Parameters":
        return self._call("hprqp_params_set_use_scaling", _bool_to_int(bool(value)))

    def set_verbose(self, value: bool) -> "Parameters":
        return self._call("hprqp_params_set_verbose", _bool_to_int(bool(value)))

    def set_spmv_mode_q(self, value: int | str) -> "Parameters":
        return self._call("hprqp_params_set_spmv_mode_q", _spmv_mode_code(value))

    def set_spmv_mode_a(self, value: int | str) -> "Parameters":
        return self._call("hprqp_params_set_spmv_mode_a", _spmv_mode_code(value))


def read_from_mps(path: str | os.PathLike[str]) -> Model:
    model = Model()
    encoded = os.fspath(path).encode("utf-8")
    _raise_for_status(_lib().hprqp_model_read_from_mps(model._handle, encoded), "hprqp_model_read_from_mps")
    return model


def build_from_mps(path: str | os.PathLike[str]) -> Model:
    return read_from_mps(path)


def _is_scipy_sparse(matrix: object) -> bool:
    return hasattr(matrix, "indptr") and hasattr(matrix, "indices") and hasattr(matrix, "data") and hasattr(matrix, "shape")


def _dense_to_csr(matrix: Sequence[Sequence[float]]) -> tuple[int, int, list[int], list[int], list[float]]:
    nrow = len(matrix)
    ncol = len(matrix[0]) if nrow > 0 else 0
    row_ptr = [0]
    col_ind: list[int] = []
    values: list[float] = []
    for row in matrix:
        if len(row) != ncol:
            raise ValueError("Dense matrix rows must all have the same length.")
        for col, value in enumerate(row):
            value_f = float(value)
            if value_f != 0.0:
                col_ind.append(col)
                values.append(value_f)
        row_ptr.append(len(col_ind))
    return nrow, ncol, row_ptr, col_ind, values


def _convert_scipy_like_to_csr(matrix: object) -> tuple[int, int, list[int], list[int], list[float]]:
    if hasattr(matrix, "tocsr"):
        csr = matrix.tocsr()
        nrow, ncol = csr.shape
        return int(nrow), int(ncol), [int(v) for v in csr.indptr], [int(v) for v in csr.indices], [float(v) for v in csr.data]
    raise TypeError("Unsupported sparse matrix object.")


def _matrix_to_csr(matrix: object) -> tuple[int, int, list[int], list[int], list[float]]:
    if _is_scipy_sparse(matrix):
        return _convert_scipy_like_to_csr(matrix)
    if isinstance(matrix, Sequence) and (len(matrix) == 0 or isinstance(matrix[0], Sequence)):
        return _dense_to_csr(matrix)  # type: ignore[arg-type]
    raise TypeError("Expected a dense row-major matrix or a SciPy sparse matrix.")


def _vector_to_list(values: Iterable[float], expected_length: int, name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != expected_length:
        raise ValueError(f"{name} length {len(result)} does not match expected length {expected_length}.")
    return result


def _size_array(values: Sequence[int]) -> ctypes.Array[ctypes.c_size_t]:
    return (ctypes.c_size_t * len(values))(*[int(value) for value in values])


def _double_array(values: Sequence[float]) -> ctypes.Array[ctypes.c_double]:
    return (ctypes.c_double * len(values))(*[float(value) for value in values])


def build_from_qabc(Q: object,
                    c: Iterable[float],
                    A: object,
                    AL: Iterable[float],
                    AU: Iterable[float],
                    l: Iterable[float],
                    u: Iterable[float],
                    obj_constant: float = 0.0) -> Model:
    q_nrow, q_ncol, q_row_ptr, q_col_ind, q_values = _matrix_to_csr(Q)
    a_nrow, a_ncol, a_row_ptr, a_col_ind, a_values = _matrix_to_csr(A)
    if q_nrow != q_ncol:
        raise ValueError("Q must be square.")
    if a_nrow == 0 and a_ncol == 0:
        a_ncol = q_ncol
    if a_ncol != q_ncol:
        raise ValueError("A column count must match Q size.")

    c_list = _vector_to_list(c, q_ncol, "c")
    l_list = _vector_to_list(l, q_ncol, "l")
    u_list = _vector_to_list(u, q_ncol, "u")
    al_list = [] if a_nrow == 0 else _vector_to_list(AL, a_nrow, "AL")
    au_list = [] if a_nrow == 0 else _vector_to_list(AU, a_nrow, "AU")

    q_row_ptr_arr = _size_array(q_row_ptr)
    q_col_ind_arr = _size_array(q_col_ind) if q_col_ind else (ctypes.c_size_t * 1)(0)
    q_values_arr = _double_array(q_values) if q_values else (ctypes.c_double * 1)(0.0)
    a_row_ptr_arr = _size_array(a_row_ptr)
    a_col_ind_arr = _size_array(a_col_ind) if a_col_ind else (ctypes.c_size_t * 1)(0)
    a_values_arr = _double_array(a_values) if a_values else (ctypes.c_double * 1)(0.0)
    c_arr = _double_array(c_list)
    l_arr = _double_array(l_list)
    u_arr = _double_array(u_list)
    al_arr = _double_array(al_list) if al_list else None
    au_arr = _double_array(au_list) if au_list else None

    model = Model()
    status = _lib().hprqp_model_build_from_qabc_csr(
        model._handle,
        q_nrow, q_ncol, len(q_values), q_row_ptr_arr, q_col_ind_arr, q_values_arr,
        a_nrow, a_ncol, len(a_values), a_row_ptr_arr, a_col_ind_arr, a_values_arr,
        c_arr,
        al_arr if al_arr is not None else None,
        au_arr if au_arr is not None else None,
        l_arr,
        u_arr,
        float(obj_constant),
    )
    _raise_for_status(status, "hprqp_model_build_from_qabc_csr")
    return model


build_from_QAbc = build_from_qabc


def _import_cvxpy() -> Any:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CVXPY support requires the optional dependency. Install it with `pip install -e .[cvxpy]`."
        ) from exc
    return cp


def _unwrap_cvxpy_problem_data(problem: object, cp: Any) -> dict[str, Any]:
    if not hasattr(problem, "get_problem_data") or not hasattr(problem, "variables") or not hasattr(problem, "objective"):
        raise TypeError("Expected a cvxpy.Problem instance.")
    if getattr(problem.objective, "NAME", "").lower() != "minimize":
        raise ValueError("Only minimization CVXPY problems are supported.")
    if hasattr(problem, "is_qp") and not problem.is_qp():
        raise ValueError("Only convex quadratic CVXPY problems are supported.")

    raw_data = problem.get_problem_data(cp.OSQP)
    data = raw_data[0] if isinstance(raw_data, tuple) else raw_data
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected CVXPY canonicalization output.")
    if data.get("bool_vars_idx") or data.get("int_vars_idx"):
        raise ValueError("Mixed-integer CVXPY problems are unsupported.")
    return data


def _evaluate_cvxpy_objective_constant(problem: object) -> float:
    import numpy as np

    variables = list(problem.variables())
    previous_values: list[Any] = []
    for variable in variables:
        value = variable.value
        previous_values.append(None if value is None else np.array(value, copy=True))

    try:
        for variable in variables:
            if variable.shape == ():
                variable.value = 0.0
            else:
                variable.value = np.zeros(variable.shape, dtype=float, order="F")
        value = problem.objective.expr.value
        if value is None:
            raise RuntimeError("Unable to evaluate the CVXPY objective constant.")
        return float(value)
    finally:
        for variable, value in zip(variables, previous_values):
            variable.value = value


def build_from_cvxpy_problem(problem: object) -> tuple[Model, CVXPYBuildInfo]:
    cp = _import_cvxpy()
    import scipy.sparse as sp

    data = _unwrap_cvxpy_problem_data(problem, cp)
    param_prob = data["param_prob"]
    variables = list(problem.variables())
    num_variables = int(data["n_var"])
    expected_num_variables = sum(int(variable.size) for variable in variables)
    if num_variables != expected_num_variables:
        raise ValueError(
            "CVXPY canonicalization introduced auxiliary variables; rewrite the model as a direct QP with explicit quadratic and affine terms."
        )

    variable_offsets = {int(key): int(value) for key, value in param_prob.var_id_to_col.items()}
    variable_shapes = {int(variable.id): tuple(int(dim) for dim in variable.shape) for variable in variables}
    for variable in variables:
        if int(variable.id) not in variable_offsets:
            raise ValueError("Unable to map a CVXPY variable back to the solver variable vector.")

    P = data["P"].tocsr()
    q = data["q"]
    A_eq = data["A"].tocsr()
    b_eq = data["b"]
    F_ineq = data["F"].tocsr()
    G_ineq = data["G"]

    rows: list[Any] = []
    AL: list[float] = []
    AU: list[float] = []
    if int(A_eq.shape[0]) > 0:
        rows.append(A_eq)
        AL.extend(float(value) for value in b_eq)
        AU.extend(float(value) for value in b_eq)
    if int(F_ineq.shape[0]) > 0:
        rows.append(F_ineq)
        AL.extend([-math.inf] * int(F_ineq.shape[0]))
        AU.extend(float(value) for value in G_ineq)

    A = sp.vstack(rows, format="csr") if rows else sp.csr_matrix((0, num_variables))
    objective_constant = _evaluate_cvxpy_objective_constant(problem)
    model = build_from_qabc(
        P,
        q,
        A,
        AL,
        AU,
        [-math.inf] * num_variables,
        [math.inf] * num_variables,
        obj_constant=objective_constant,
    )
    return model, CVXPYBuildInfo(
        variable_offsets=variable_offsets,
        variable_shapes=variable_shapes,
        num_variables=num_variables,
        num_constraints=int(A.shape[0]),
        objective_constant=objective_constant,
    )


def _assign_cvxpy_variable_values(problem: object, info: CVXPYBuildInfo, solution: Sequence[float]) -> None:
    import numpy as np

    for variable in problem.variables():
        start = info.variable_offsets[int(variable.id)]
        stop = start + int(variable.size)
        values = np.asarray(solution[start:stop], dtype=float)
        if variable.shape == ():
            variable.value = float(values[0])
        else:
            variable.value = values.reshape(variable.shape, order="F")


def solve_cvxpy_problem(problem: object,
                        params: Parameters | None = None,
                        assign_values: bool = True) -> Result:
    model, info = build_from_cvxpy_problem(problem)
    result = optimize(model, params)
    if assign_values:
        _assign_cvxpy_variable_values(problem, info, result.x)
    return result


def _copy_vector(pointer: ctypes.POINTER(ctypes.c_double), length: int) -> list[float]:
    if not pointer or length == 0:
        return []
    return [float(pointer[index]) for index in range(length)]


def optimize(model: Model, params: Parameters | None = None) -> Result:
    if params is None:
        params = Parameters()

    results_handle = ctypes.c_void_p()
    _raise_for_status(_lib().hprqp_results_create(ctypes.byref(results_handle)), "hprqp_results_create")
    try:
        status_code = int(_lib().hprqp_optimize(model._handle, params._handle, results_handle))
        n = int(_lib().hprqp_results_num_variables(results_handle))
        m = int(_lib().hprqp_results_num_constraints(results_handle))
        return Result(
            status_code=status_code,
            status=status_name(status_code),
            iter=int(_lib().hprqp_results_iter(results_handle)),
            iter_4=int(_lib().hprqp_results_iter_4(results_handle)),
            iter_6=int(_lib().hprqp_results_iter_6(results_handle)),
            time=float(_lib().hprqp_results_time(results_handle)),
            time_4=float(_lib().hprqp_results_time_4(results_handle)),
            time_6=float(_lib().hprqp_results_time_6(results_handle)),
            objective=float(_lib().hprqp_results_objective(results_handle)),
            residual=float(_lib().hprqp_results_residual(results_handle)),
            gap=float(_lib().hprqp_results_gap(results_handle)),
            x=_copy_vector(_lib().hprqp_results_x_data(results_handle), n),
            y=_copy_vector(_lib().hprqp_results_y_data(results_handle), m),
            z=_copy_vector(_lib().hprqp_results_z_data(results_handle), n),
            w=_copy_vector(_lib().hprqp_results_w_data(results_handle), n),
        )
    finally:
        _lib().hprqp_results_destroy(results_handle)


__all__ = [
    "CVXPYBuildInfo",
    "Model",
    "Parameters",
    "Result",
    "SPMV_MODE_AUTO",
    "SPMV_MODE_CUSPARSE",
    "SPMV_MODE_CUSTOMIZED",
    "STATUS_ALLOCATION_FAILED",
    "STATUS_CONTINUE",
    "STATUS_INVALID_ARGUMENT",
    "STATUS_MAX_ITER",
    "STATUS_NOT_IMPLEMENTED",
    "STATUS_OK",
    "STATUS_OPTIMAL",
    "STATUS_TIME_LIMIT",
    "build_from_cvxpy_problem",
    "build_from_mps",
    "build_from_QAbc",
    "build_from_qabc",
    "gpu_backend_available",
    "optimize",
    "read_from_mps",
    "solve_cvxpy_problem",
    "status_name",
]