using SparseArrays
using HPRQP

println("GPU backend available: ", gpu_backend_available())

Q = sparse([2.0 0.0; 0.0 4.0])
A = sparse([1.0 1.0])
c = [2.0, 1.0]
AL = [2.0]
AU = [2.0]
l = [0.0, 0.0]
u = [10.0, 10.0]

model = build_from_QAbc(Q, c, A, AL, AU, l, u)
params = Parameters(use_gpu = true, device_number = 0, verbose = false, stoptol = 1e-8)
result = optimize(model, params)

println("status = ", result.status)
println("objective = ", result.objective)
println("x = ", result.x)
println("y = ", result.y)