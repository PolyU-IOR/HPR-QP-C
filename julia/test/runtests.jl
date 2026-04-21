using Test
using SparseArrays
using JuMP
using HPRQP
import MathOptInterface as MOI

@testset "HPRQP Julia Wrapper" begin
    @test status_name(HPRQP.STATUS_OPTIMAL) == "OPTIMAL"

    mps_path = normpath(joinpath(@__DIR__, "..", "..", "tests", "data", "simple_qp.mps"))
    model = read_from_mps(mps_path)
    params = Parameters(verbose=false, max_iter=0)
    result = optimize(model, params)

    @test result.status == :max_iter
    @test result.status_code == HPRQP.STATUS_MAX_ITER
    @test length(result.x) == 2
    @test length(result.y) == 2

    Q = sparse([2.0 0.0; 0.0 2.0])
    A = spzeros(0, 2)
    c = [-4.0, -2.0]
    AL = Float64[]
    AU = Float64[]
    l = [-Inf, -Inf]
    u = [Inf, Inf]

    model_qabc = build_from_QAbc(Q, c, A, AL, AU, l, u)
    params_qabc = Parameters(verbose=false, stoptol=1e-8)
    result_qabc = optimize(model_qabc, params_qabc)

    @test result_qabc.status == :optimal
    @test isapprox(result_qabc.objective, -5.0; atol=1e-6, rtol=1e-6)
    @test isapprox(result_qabc.x[1], 2.0; atol=1e-6, rtol=1e-6)
    @test isapprox(result_qabc.x[2], 1.0; atol=1e-6, rtol=1e-6)
    @test isempty(result_qabc.y)

    @testset "GPU Path" begin
        if gpu_backend_available()
            Qg = sparse([2.0 0.0; 0.0 4.0])
            Ag = sparse([1.0 1.0])
            cg = [2.0, 1.0]
            ALg = [2.0]
            AUg = [2.0]
            lg = [0.0, 0.0]
            ug = [10.0, 10.0]

            gpu_model = build_from_QAbc(Qg, cg, Ag, ALg, AUg, lg, ug)
            cpu_params = Parameters(verbose=false, use_gpu=false)
            gpu_params = Parameters(verbose=false, use_gpu=true, device_number=0)
            cpu_result = optimize(gpu_model, cpu_params)
            gpu_result = optimize(gpu_model, gpu_params)

            @test cpu_result.status == :optimal
            @test gpu_result.status == :optimal
            @test isapprox(cpu_result.objective, gpu_result.objective; atol=1e-6, rtol=1e-6)
            @test isapprox(cpu_result.residual, gpu_result.residual; atol=1e-6, rtol=1e-6)
            @test isapprox(cpu_result.gap, gpu_result.gap; atol=1e-6, rtol=1e-6)
            @test isapprox(cpu_result.x[1], gpu_result.x[1]; atol=1e-6, rtol=1e-6)
            @test isapprox(cpu_result.x[2], gpu_result.x[2]; atol=1e-6, rtol=1e-6)
        end
    end

    @testset "JuMP Optimizer" begin
        model = JuMP.Model(HPRQP.Optimizer)
        set_optimizer_attribute(model, "use_gpu", true)
        set_optimizer_attribute(model, "device_number", 0)
        set_optimizer_attribute(model, "verbose", false)
        set_optimizer_attribute(model, "stoptol", 1e-8)

        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @constraint(model, x + y == 2)
        @objective(model, Min, x^2 + 2y^2 + 2x + y)

        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test isapprox(objective_value(model), 71.0 / 12.0; atol=1e-6, rtol=1e-6)
        @test isapprox(value(x), 7.0 / 6.0; atol=1e-6, rtol=1e-6)
        @test isapprox(value(y), 5.0 / 6.0; atol=1e-6, rtol=1e-6)
    end
end