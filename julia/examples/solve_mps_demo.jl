using HPRQP

mps_path = isempty(ARGS) ? normpath(joinpath(@__DIR__, "..", "..", "tests", "data", "simple_qp.mps")) : ARGS[1]

println("Solving MPS: ", mps_path)
println("GPU backend available: ", gpu_backend_available())

model = read_from_mps(mps_path)
params = Parameters(use_gpu = false, device_number = 0, stoptol = 1e-8)
result = optimize(model, params)

println("status = ", result.status)
println("objective = ", result.objective)
println("residual = ", result.residual)
println("gap = ", result.gap)
println("iter = ", result.iter)
# println("x = ", result.x)
# println("y = ", result.y)