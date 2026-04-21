const _SUPPORTED_RAW_PARAMETERS = Set([
    "stoptol",
    "max_iter",
    "use_gpu",
    "device_number",
    "time_limit",
    "check_iter",
    "print_frequency",
    "use_scaling",
    "verbose",
    "spmv_mode_q",
    "spmv_mode_a",
])

function _default_parameter_values()
    return Dict{String, Any}(
        "stoptol" => 1e-8,
        "max_iter" => typemax(Int),
        "use_gpu" => true,
        "device_number" => 0,
        "time_limit" => 3600.0,
        "check_iter" => 100,
        "print_frequency" => -1,
        "use_scaling" => true,
        "verbose" => true,
        "spmv_mode_q" => :auto,
        "spmv_mode_a" => :auto,
    )
end

const _SCALAR_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

MOI.Utilities.@product_of_sets(
    _HPRQPSets,
    MOI.EqualTo{T},
    MOI.LessThan{T},
    MOI.GreaterThan{T},
    MOI.Interval{T},
)

const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{Float64, Int, MOI.Utilities.OneBasedIndexing},
        MOI.Utilities.Hyperrectangle{Float64},
        _HPRQPSets{Float64},
    },
}

mutable struct Optimizer <: MOI.AbstractOptimizer
    params::Parameters
    parameter_values::Dict{String, Any}
    results::Union{Nothing, Result}
    silent::Bool
    cache::Union{Nothing, OptimizerCache}
    index_map::Union{Nothing, MOI.IndexMap}
    obj_sense::MOI.OptimizationSense

    function Optimizer()
        return new(Parameters(), _default_parameter_values(), nothing, false, nothing, nothing, MOI.MIN_SENSE)
    end
end

function MOI.default_cache(::Optimizer, ::Type{Float64})
    return MOI.Utilities.UniversalFallback(OptimizerCache())
end

function MOI.is_empty(model::Optimizer)
    return model.cache === nothing && model.results === nothing
end

function MOI.empty!(model::Optimizer)
    model.results = nothing
    model.cache = nothing
    model.index_map = nothing
    model.obj_sense = MOI.MIN_SENSE
    return nothing
end

MOI.get(::Optimizer, ::MOI.SolverName) = "HPRQP"
MOI.get(::Optimizer, ::MOI.SolverVersion) = "0.1.0"
MOI.supports_incremental_interface(::Optimizer) = false

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return nothing
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    set_time_limit!(model.params, value)
    model.parameter_values["time_limit"] = Float64(value)
    return nothing
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    set_time_limit!(model.params, 3600.0)
    model.parameter_values["time_limit"] = 3600.0
    return nothing
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return Float64(model.parameter_values["time_limit"])
end

MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = false

function MOI.supports(::Optimizer, attr::MOI.RawOptimizerAttribute)
    return attr.name in _SUPPORTED_RAW_PARAMETERS
end

function _set_raw_parameter!(model::Optimizer, name::String, value)
    if name == "stoptol"
        set_stoptol!(model.params, value)
    elseif name == "max_iter"
        set_max_iter!(model.params, value)
    elseif name == "use_gpu"
        set_use_gpu!(model.params, Bool(value))
    elseif name == "device_number"
        set_device_number!(model.params, value)
    elseif name == "time_limit"
        set_time_limit!(model.params, value)
    elseif name == "check_iter"
        set_check_iter!(model.params, value)
    elseif name == "print_frequency"
        set_print_frequency!(model.params, value)
    elseif name == "use_scaling"
        set_use_scaling!(model.params, Bool(value))
    elseif name == "verbose"
        set_verbose!(model.params, Bool(value))
    elseif name == "spmv_mode_q"
        set_spmv_mode_q!(model.params, value)
    elseif name == "spmv_mode_a"
        set_spmv_mode_a!(model.params, value)
    else
        throw(MOI.UnsupportedAttribute(MOI.RawOptimizerAttribute(name)))
    end
    model.parameter_values[name] = value
    return nothing
end

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    _set_raw_parameter!(model, String(attr.name), value)
    return nothing
end

function MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    name = String(attr.name)
    if !haskey(model.parameter_values, name)
        throw(MOI.UnsupportedAttribute(attr))
    end
    return model.parameter_values[name]
end

function MOI.supports_constraint(::Optimizer,
                                 ::Type{<:Union{MOI.VariableIndex, MOI.ScalarAffineFunction{Float64}}},
                                 ::Type{<:_SCALAR_SETS})
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}) = true

function MOI.copy_to(dest::Optimizer, src::OptimizerCache)
    dest.cache = src
    dest.results = nothing
    dest.index_map = _build_index_map(src)
    return dest.index_map
end

function MOI.copy_to(dest::Optimizer, src::MOI.Utilities.UniversalFallback{OptimizerCache})
    MOI.Utilities.throw_unsupported(src)
    return MOI.copy_to(dest, src.model)
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    MOI.copy_to(dest, cache)
    return index_map
end

function _build_index_map(src::OptimizerCache)
    index_map = MOI.IndexMap()
    for (i, x) in enumerate(MOI.get(src, MOI.ListOfVariableIndices()))
        index_map[x] = MOI.VariableIndex(i)
    end
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        _index_map_constraints(src, index_map, F, S)
    end
    return index_map
end

function _index_map_constraints(src::OptimizerCache,
                                index_map,
                                ::Type{MOI.ScalarAffineFunction{Float64}},
                                ::Type{S}) where {S<:_SCALAR_SETS}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, S}())
        row = MOI.Utilities.rows(src.constraints, ci)
        index_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}(row)
    end
    return nothing
end

function _index_map_constraints(src::OptimizerCache,
                                index_map,
                                ::Type{MOI.VariableIndex},
                                ::Type{S}) where {S<:_SCALAR_SETS}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex, S}())
        col = index_map[MOI.VariableIndex(ci.value)].value
        index_map[ci] = MOI.ConstraintIndex{MOI.VariableIndex, S}(col)
    end
    return nothing
end

function _quadratic_objective_to_qc(obj::MOI.ScalarQuadraticFunction{Float64}, n::Int)
    c = zeros(Float64, n)
    for term in obj.affine_terms
        c[term.variable.value] += term.coefficient
    end

    row_index = Int[]
    col_index = Int[]
    values = Float64[]
    for term in obj.quadratic_terms
        i = term.variable_1.value
        j = term.variable_2.value
        coeff = term.coefficient
        if i == j
            push!(row_index, i)
            push!(col_index, j)
            push!(values, coeff)
        else
            push!(row_index, i)
            push!(col_index, j)
            push!(values, 2.0 * coeff)
            push!(row_index, j)
            push!(col_index, i)
            push!(values, 2.0 * coeff)
        end
    end

    Q = isempty(values) ? spzeros(n, n) : sparse(row_index, col_index, values, n, n)
    return Q, c, obj.constant
end

function _affine_objective_to_qc(obj::MOI.ScalarAffineFunction{Float64}, n::Int)
    c = zeros(Float64, n)
    for term in obj.terms
        c[term.variable.value] += term.coefficient
    end
    return spzeros(n, n), c, obj.constant
end

function _apply_silent_mode!(model::Optimizer)
    verbose_flag = Bool(get(model.parameter_values, "verbose", true))
    set_verbose!(model.params, model.silent ? false : verbose_flag)
    return nothing
end

function MOI.optimize!(model::Optimizer, src::OptimizerCache)
    A = src.constraints.coefficients
    row_bounds = src.constraints.constants
    n = A.n

    obj_type = MOI.get(src, MOI.ObjectiveFunctionType())
    Q = spzeros(n, n)
    c = zeros(Float64, n)
    obj_constant = 0.0
    if obj_type == MOI.ScalarQuadraticFunction{Float64}
        Q, c, obj_constant = _quadratic_objective_to_qc(MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}()), n)
    elseif obj_type == MOI.ScalarAffineFunction{Float64}
        Q, c, obj_constant = _affine_objective_to_qc(MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()), n)
    else
        error("Unsupported objective function type: $(obj_type)")
    end

    model.obj_sense = MOI.get(src, MOI.ObjectiveSense())
    if model.obj_sense == MOI.MAX_SENSE
        Q = -Q
        c = -c
        obj_constant = -obj_constant
    end

    l = Float64.(src.variables.lower)
    u = Float64.(src.variables.upper)
    AL = Float64.(row_bounds.lower)
    AU = Float64.(row_bounds.upper)
    A_sparse = SparseMatrixCSC{Float64, Int}(A.m, A.n, copy(A.colptr), copy(A.rowval), copy(A.nzval))
    Q_sparse = SparseMatrixCSC{Float64, Int}(Q)

    _apply_silent_mode!(model)
    low_level_model = build_from_QAbc(Q_sparse, c, A_sparse, AL, AU, l, u; obj_constant=obj_constant)
    model.results = optimize(low_level_model, model.params)
    return nothing
end

function MOI.optimize!(model::Optimizer)
    if model.cache === nothing
        error("No model has been loaded. Build a JuMP model first.")
    end
    MOI.optimize!(model, model.cache)
    return nothing
end

function MOI.get(model::Optimizer, ::MOI.SolveTimeSec)
    return model.results === nothing ? 0.0 : model.results.time
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available. Call optimize! first.")
    end
    value = model.results.objective
    return model.obj_sense == MOI.MAX_SENSE ? -value : value
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.results === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    elseif model.results.status == :optimal
        return MOI.OPTIMAL
    elseif model.results.status == :max_iter
        return MOI.ITERATION_LIMIT
    elseif model.results.status == :time_limit
        return MOI.TIME_LIMIT
    elseif model.results.status == :invalid_argument
        return MOI.INVALID_MODEL
    else
        return MOI.OTHER_ERROR
    end
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    return model.results === nothing ? "OPTIMIZE_NOT_CALLED" : status_name(model.results.status_code)
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.results === nothing ? 0 : 1
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1 || model.results === nothing
        return MOI.NO_SOLUTION
    end
    if model.results.status in (:optimal, :max_iter, :time_limit)
        return MOI.FEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if attr.result_index != 1 || model.results === nothing
        return MOI.NO_SOLUTION
    end
    if model.results.status in (:optimal, :max_iter, :time_limit)
        return MOI.FEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    if model.cache !== nothing
        return length(MOI.get(model.cache, MOI.ListOfVariableIndices()))
    elseif model.results !== nothing
        return length(model.results.x)
    end
    return 0
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, x::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    return model.results.x[x.value]
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, xs::Vector{MOI.VariableIndex})
    MOI.check_result_index_bounds(model, attr)
    return [MOI.get(model, attr, x) for x in xs]
end

function MOI.get(model::Optimizer,
                 attr::MOI.ConstraintPrimal,
                 c::MOI.ConstraintIndex{MOI.VariableIndex, <: _SCALAR_SETS})
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    return model.results.x[c.value]
end

function MOI.get(model::Optimizer,
                 attr::MOI.ConstraintDual,
                 c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, <: _SCALAR_SETS})
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing || c.value > length(model.results.y)
        error("No row-dual result available.")
    end
    dual_value = model.results.y[c.value]
    return model.obj_sense == MOI.MAX_SENSE ? -dual_value : dual_value
end

function MOI.get(model::Optimizer,
                 attr::MOI.ConstraintDual,
                 c::MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{Float64}})
    MOI.check_result_index_bounds(model, attr)
    dual_value = min(0.0, model.results.z[c.value])
    return model.obj_sense == MOI.MAX_SENSE ? -dual_value : dual_value
end

function MOI.get(model::Optimizer,
                 attr::MOI.ConstraintDual,
                 c::MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{Float64}})
    MOI.check_result_index_bounds(model, attr)
    dual_value = max(0.0, model.results.z[c.value])
    return model.obj_sense == MOI.MAX_SENSE ? -dual_value : dual_value
end

function MOI.get(model::Optimizer,
                 attr::MOI.ConstraintDual,
                 c::MOI.ConstraintIndex{MOI.VariableIndex, <:Union{MOI.Interval{Float64}, MOI.EqualTo{Float64}}})
    MOI.check_result_index_bounds(model, attr)
    dual_value = model.results.z[c.value]
    return model.obj_sense == MOI.MAX_SENSE ? -dual_value : dual_value
end