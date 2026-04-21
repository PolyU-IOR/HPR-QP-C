using JuMP
using HPRQP

println("GPU backend available: ", gpu_backend_available())

model = JuMP.Model(HPRQP.Optimizer)
set_optimizer_attribute(model, "use_gpu", true)
set_optimizer_attribute(model, "device_number", 0)
set_optimizer_attribute(model, "stoptol", 1e-8)
set_optimizer_attribute(model, "verbose", false)

@variable(model, x >= 0)
@variable(model, y >= 0)
@constraint(model, x + y == 2)
@objective(model, Min, x^2 + 2y^2 + 2x + y)

optimize!(model)

println("termination_status = ", termination_status(model))
println("objective_value = ", objective_value(model))
println("x = ", value(x))
println("y = ", value(y))