module HPRQP

using Libdl
using SparseArrays
import MathOptInterface as MOI

export Model
export Optimizer
export Parameters
export Result
export build_from_QAbc
export build_from_mps
export gpu_backend_available
export optimize
export read_from_mps
export set_check_iter!
export set_device_number!
export set_max_iter!
export set_print_frequency!
export set_spmv_mode_a!
export set_spmv_mode_q!
export set_stoptol!
export set_time_limit!
export set_use_gpu!
export set_use_scaling!
export set_verbose!
export status_name

const _LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _LIB_PATH = Ref{String}("")

const STATUS_OK = 0
const STATUS_INVALID_ARGUMENT = 1
const STATUS_ALLOCATION_FAILED = 2
const STATUS_NOT_IMPLEMENTED = 3
const STATUS_MAX_ITER = 4
const STATUS_TIME_LIMIT = 5
const STATUS_OPTIMAL = 6
const STATUS_CONTINUE = 7

const SPMV_MODE_AUTO = 0
const SPMV_MODE_CUSPARSE = 1
const SPMV_MODE_CUSTOMIZED = 2

mutable struct Model
    handle::Ptr{Cvoid}
end

mutable struct Parameters
    handle::Ptr{Cvoid}
end

struct Result
    status_code::Int32
    status::Symbol
    iter::Int
    iter_4::Int
    iter_6::Int
    time::Float64
    time_4::Float64
    time_6::Float64
    objective::Float64
    residual::Float64
    gap::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    w::Vector{Float64}
end

function _package_root()
    return normpath(joinpath(@__DIR__, "..", ".."))
end

function _library_candidates()
    repo_root = _package_root()
    names = Sys.iswindows() ? ["hprqp.dll", "libhprqp.dll"] : Sys.isapple() ? ["libhprqp.dylib"] : ["libhprqp.so"]
    dirs = String[]

    if haskey(ENV, "HPRQP_LIBRARY_PATH")
        push!(dirs, ENV["HPRQP_LIBRARY_PATH"])
    end

    append!(dirs, [
        joinpath(repo_root, ".install-gpu-test", "lib"),
        joinpath(repo_root, ".install-cpu-test", "lib"),
        joinpath(repo_root, "build-cuda-shared"),
        joinpath(repo_root, "build-shared"),
        joinpath(repo_root, "build-cuda"),
        joinpath(repo_root, "build"),
    ])

    candidates = String[]
    for dir in dirs
        if isempty(dir)
            continue
        end
        if isfile(dir)
            push!(candidates, dir)
            continue
        end
        for name in names
            push!(candidates, joinpath(dir, name))
        end
    end

    return candidates
end

function _ensure_library_loaded()
    if _LIB_HANDLE[] != C_NULL
        return
    end

    for candidate in _library_candidates()
        if !isfile(candidate)
            continue
        end
        handle = Libdl.dlopen_e(candidate)
        if handle != C_NULL
            _LIB_HANDLE[] = handle
            _LIB_PATH[] = candidate
            return
        end
    end

    error("Unable to locate libhprqp. Build it first with `make gpu-build-shared` or `make build-shared`, or set HPRQP_LIBRARY_PATH to the shared library path.")
end

function __init__()
    _ensure_library_loaded()
end

function _sym(name::Symbol)
    _ensure_library_loaded()
    return Libdl.dlsym(_LIB_HANDLE[], name)
end

function _destroy_model!(model::Model)
    if model.handle != C_NULL
        ccall(_sym(:hprqp_model_destroy), Cvoid, (Ptr{Cvoid},), model.handle)
        model.handle = C_NULL
    end
    return nothing
end

function _destroy_params!(params::Parameters)
    if params.handle != C_NULL
        ccall(_sym(:hprqp_params_destroy), Cvoid, (Ptr{Cvoid},), params.handle)
        params.handle = C_NULL
    end
    return nothing
end

function _create_model_from_handle(handle::Ptr{Cvoid})
    model = Model(handle)
    finalizer(_destroy_model!, model)
    return model
end

function _create_params_from_handle(handle::Ptr{Cvoid})
    params = Parameters(handle)
    finalizer(_destroy_params!, params)
    return params
end

function _check_status(code::Integer, context::AbstractString)
    status = Int32(code)
    if status != STATUS_OK
        error("$(context) failed with status $(status_name(status))")
    end
    return status
end

function status_name(status::Integer)
    ptr = ccall(_sym(:hprqp_status_name), Cstring, (Cint,), Cint(status))
    return unsafe_string(ptr)
end

function _status_symbol(status::Integer)
    name = lowercase(status_name(status))
    return Symbol(name)
end

function gpu_backend_available()
    return ccall(_sym(:hprqp_gpu_backend_available), Cint, ()) != 0
end

function Model()
    out = Ref{Ptr{Cvoid}}(C_NULL)
    _check_status(ccall(_sym(:hprqp_model_create), Cint, (Ref{Ptr{Cvoid}},), out), "hprqp_model_create")
    return _create_model_from_handle(out[])
end

function _create_parameters()
    out = Ref{Ptr{Cvoid}}(C_NULL)
    _check_status(ccall(_sym(:hprqp_params_create), Cint, (Ref{Ptr{Cvoid}},), out), "hprqp_params_create")
    return _create_params_from_handle(out[])
end

function Parameters(; kwargs...)
    params = _create_parameters()
    for (name, value) in kwargs
        _apply_parameter!(params, name, value)
    end
    return params
end

function _apply_parameter!(params::Parameters, name::Symbol, value)
    if name === :stoptol
        return set_stoptol!(params, value)
    elseif name === :max_iter
        return set_max_iter!(params, value)
    elseif name === :use_gpu
        return set_use_gpu!(params, value)
    elseif name === :device_number
        return set_device_number!(params, value)
    elseif name === :time_limit
        return set_time_limit!(params, value)
    elseif name === :check_iter
        return set_check_iter!(params, value)
    elseif name === :print_frequency
        return set_print_frequency!(params, value)
    elseif name === :use_scaling
        return set_use_scaling!(params, value)
    elseif name === :verbose
        return set_verbose!(params, value)
    elseif name === :spmv_mode_q
        return set_spmv_mode_q!(params, value)
    elseif name === :spmv_mode_a
        return set_spmv_mode_a!(params, value)
    end
    throw(ArgumentError("Unsupported parameter keyword: $(name)"))
end

function _set_double!(symbol::Symbol, params::Parameters, value::Real)
    _check_status(ccall(_sym(symbol), Cint, (Ptr{Cvoid}, Cdouble), params.handle, Float64(value)), String(symbol))
    return params
end

function _set_size_t!(symbol::Symbol, params::Parameters, value::Integer)
    _check_status(ccall(_sym(symbol), Cint, (Ptr{Cvoid}, Csize_t), params.handle, Csize_t(value)), String(symbol))
    return params
end

function _set_int!(symbol::Symbol, params::Parameters, value::Integer)
    _check_status(ccall(_sym(symbol), Cint, (Ptr{Cvoid}, Cint), params.handle, Cint(value)), String(symbol))
    return params
end

function _set_bool!(symbol::Symbol, params::Parameters, value)
    return _set_int!(symbol, params, value ? 1 : 0)
end

function set_stoptol!(params::Parameters, value::Real)
    return _set_double!(:hprqp_params_set_stoptol, params, value)
end

function set_max_iter!(params::Parameters, value::Integer)
    return _set_size_t!(:hprqp_params_set_max_iter, params, value)
end

function set_use_gpu!(params::Parameters, value::Bool)
    return _set_bool!(:hprqp_params_set_use_gpu, params, value)
end

function set_device_number!(params::Parameters, value::Integer)
    return _set_int!(:hprqp_params_set_device_number, params, value)
end

function set_time_limit!(params::Parameters, value::Real)
    return _set_double!(:hprqp_params_set_time_limit, params, value)
end

function set_check_iter!(params::Parameters, value::Integer)
    return _set_size_t!(:hprqp_params_set_check_iter, params, value)
end

function set_print_frequency!(params::Parameters, value::Integer)
    return _set_int!(:hprqp_params_set_print_frequency, params, value)
end

function set_use_scaling!(params::Parameters, value::Bool)
    return _set_bool!(:hprqp_params_set_use_scaling, params, value)
end

function set_verbose!(params::Parameters, value::Bool)
    return _set_bool!(:hprqp_params_set_verbose, params, value)
end

function _spmv_mode_code(value)
    if value in (:auto, "auto")
        return SPMV_MODE_AUTO
    elseif value in (:cusparse, "cusparse")
        return SPMV_MODE_CUSPARSE
    elseif value in (:customized, "customized")
        return SPMV_MODE_CUSTOMIZED
    end
    throw(ArgumentError("Unsupported SpMV mode: $(value). Expected :auto, :cusparse, or :customized."))
end

function set_spmv_mode_q!(params::Parameters, value)
    return _set_int!(:hprqp_params_set_spmv_mode_q, params, _spmv_mode_code(value))
end

function set_spmv_mode_a!(params::Parameters, value)
    return _set_int!(:hprqp_params_set_spmv_mode_a, params, _spmv_mode_code(value))
end

function read_from_mps(path::AbstractString)
    model = Model()
    status = ccall(_sym(:hprqp_model_read_from_mps), Cint, (Ptr{Cvoid}, Cstring), model.handle, path)
    _check_status(status, "hprqp_model_read_from_mps")
    return model
end

const build_from_mps = read_from_mps

function _to_sparse_float64(matrix)
    if isa(matrix, SparseMatrixCSC{Float64})
        return matrix
    elseif isa(matrix, SparseMatrixCSC)
        return SparseMatrixCSC{Float64, Int}(matrix)
    elseif isa(matrix, AbstractMatrix)
        return sparse(Float64.(matrix))
    end
    throw(ArgumentError("Expected a sparse or dense matrix."))
end

function _to_float64_vector(values, expected_length::Integer, name::AbstractString)
    vector = Float64.(collect(values))
    if length(vector) != expected_length
        throw(DimensionMismatch("$(name) length $(length(vector)) does not match expected length $(expected_length)."))
    end
    return vector
end

function _csc_to_csr(matrix::SparseMatrixCSC{Float64, <:Integer})
    nrow, ncol = size(matrix)
    nnz_count = nnz(matrix)
    row_ptr = zeros(Csize_t, nrow + 1)
    col_ind = Vector{Csize_t}(undef, nnz_count)
    values = Vector{Float64}(undef, nnz_count)

    for row in rowvals(matrix)
        row_ptr[row + 1] += 1
    end
    for i in 1:nrow
        row_ptr[i + 1] += row_ptr[i]
    end

    next_pos = copy(row_ptr[1:nrow])
    for col in 1:ncol
        for idx in nzrange(matrix, col)
            row = rowvals(matrix)[idx]
            pos = Int(next_pos[row] + 1)
            col_ind[pos] = Csize_t(col - 1)
            values[pos] = nonzeros(matrix)[idx]
            next_pos[row] += 1
        end
    end

    return row_ptr, col_ind, values
end

function build_from_QAbc(Q,
                         c,
                         A,
                         AL,
                         AU,
                         l,
                         u;
                         obj_constant::Real = 0.0)
    q_sparse = _to_sparse_float64(Q)
    a_sparse = _to_sparse_float64(A)

    n = size(q_sparse, 2)
    m = size(a_sparse, 1)

    if size(q_sparse, 1) != n
        throw(DimensionMismatch("Q must be square."))
    end
    if size(a_sparse, 2) != n
        throw(DimensionMismatch("A column count must match Q size."))
    end

    c_vec = _to_float64_vector(c, n, "c")
    l_vec = _to_float64_vector(l, n, "l")
    u_vec = _to_float64_vector(u, n, "u")
    al_vec = m == 0 ? Float64[] : _to_float64_vector(AL, m, "AL")
    au_vec = m == 0 ? Float64[] : _to_float64_vector(AU, m, "AU")

    q_row_ptr, q_col_ind, q_values = _csc_to_csr(q_sparse)
    a_row_ptr, a_col_ind, a_values = _csc_to_csr(a_sparse)

    model = Model()
    GC.@preserve q_row_ptr q_col_ind q_values a_row_ptr a_col_ind a_values c_vec al_vec au_vec l_vec u_vec begin
        status = ccall(_sym(:hprqp_model_build_from_qabc_csr),
                       Cint,
                       (Ptr{Cvoid},
                        Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Cdouble},
                        Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Cdouble},
                        Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble),
                       model.handle,
                       Csize_t(size(q_sparse, 1)), Csize_t(size(q_sparse, 2)), Csize_t(length(q_values)), q_row_ptr, q_col_ind, q_values,
                       Csize_t(size(a_sparse, 1)), Csize_t(size(a_sparse, 2)), Csize_t(length(a_values)), a_row_ptr, a_col_ind, a_values,
                       c_vec,
                       m == 0 ? C_NULL : pointer(al_vec),
                       m == 0 ? C_NULL : pointer(au_vec),
                       l_vec,
                       u_vec,
                       Float64(obj_constant))
        _check_status(status, "hprqp_model_build_from_qabc_csr")
    end
    return model
end

function _copy_result_vector(ptr::Ptr{Cdouble}, n::Int)
    if ptr == C_NULL || n == 0
        return Float64[]
    end
    return unsafe_wrap(Vector{Float64}, ptr, n) |> copy
end

function optimize(model::Model, params::Parameters = Parameters())
    results_ref = Ref{Ptr{Cvoid}}(C_NULL)
    _check_status(ccall(_sym(:hprqp_results_create), Cint, (Ref{Ptr{Cvoid}},), results_ref), "hprqp_results_create")
    raw_results = results_ref[]

    try
        status = ccall(_sym(:hprqp_optimize), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), model.handle, params.handle, raw_results)
        status_code = Int32(status)
        n = Int(ccall(_sym(:hprqp_results_num_variables), Csize_t, (Ptr{Cvoid},), raw_results))
        m = Int(ccall(_sym(:hprqp_results_num_constraints), Csize_t, (Ptr{Cvoid},), raw_results))

        return Result(
            status_code,
            _status_symbol(status_code),
            Int(ccall(_sym(:hprqp_results_iter), Csize_t, (Ptr{Cvoid},), raw_results)),
            Int(ccall(_sym(:hprqp_results_iter_4), Csize_t, (Ptr{Cvoid},), raw_results)),
            Int(ccall(_sym(:hprqp_results_iter_6), Csize_t, (Ptr{Cvoid},), raw_results)),
            ccall(_sym(:hprqp_results_time), Cdouble, (Ptr{Cvoid},), raw_results),
            ccall(_sym(:hprqp_results_time_4), Cdouble, (Ptr{Cvoid},), raw_results),
            ccall(_sym(:hprqp_results_time_6), Cdouble, (Ptr{Cvoid},), raw_results),
            ccall(_sym(:hprqp_results_objective), Cdouble, (Ptr{Cvoid},), raw_results),
            ccall(_sym(:hprqp_results_residual), Cdouble, (Ptr{Cvoid},), raw_results),
            ccall(_sym(:hprqp_results_gap), Cdouble, (Ptr{Cvoid},), raw_results),
            _copy_result_vector(ccall(_sym(:hprqp_results_x_data), Ptr{Cdouble}, (Ptr{Cvoid},), raw_results), n),
            _copy_result_vector(ccall(_sym(:hprqp_results_y_data), Ptr{Cdouble}, (Ptr{Cvoid},), raw_results), m),
            _copy_result_vector(ccall(_sym(:hprqp_results_z_data), Ptr{Cdouble}, (Ptr{Cvoid},), raw_results), n),
            _copy_result_vector(ccall(_sym(:hprqp_results_w_data), Ptr{Cdouble}, (Ptr{Cvoid},), raw_results), n),
        )
    finally
        ccall(_sym(:hprqp_results_destroy), Cvoid, (Ptr{Cvoid},), raw_results)
    end
end

include("MOI_wrapper.jl")

end