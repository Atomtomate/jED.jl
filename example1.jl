using Pkg
Pkg.activate(@__DIR__)
using jED
using TimerOutputs
using LinearAlgebra

to = TimerOutput()

@timeit to "Basis" basis = jED.Basis(3)
ϵₖ = [0.3, 0.4]#, 0.5, 0.6]#, 0.7, 0.8]#, 0.9]
Vₖ = [0.6, 0.7]#, 0.8, 0.9]#, 1.0, 1.1]#, 1.2]
U  = 1.0
μ  = 0.5
β  = 10.0
@timeit to "Model" model = AIM(ϵₖ, Vₖ, μ, U)
@timeit to "Eigen problem" es = Eigenspace(model, basis);
Z = calc_Z(es, β)
E = calc_E(es, β)
println("E₀ = $(es.E0)\nZ  = $Z\nE  = $E")
