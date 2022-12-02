using Pkg
Pkg.activate(@__DIR__)
using jED
using TimerOutputs
using LinearAlgebra

to = TimerOutput()

# @timeit to "Basis" basis = jED.Basis(7)
# ϵₖ = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# Vₖ = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
@timeit to "Basis" basis = jED.Basis(5)
#ϵₖ = [0.24379126936809078, 1.1042333108436919, -0.24379126936809078, -1.1042333108436919]
#Vₖ = [0.25917560591147798, 0.23335139853990977, 0.25917560591147798,  0.23335139853990977]
#ϵₖ = [0.23720747964147854, 1.0761589467007526, -0.23720747964147854, -1.076158946700752]
#Vₖ = [0.25043951279646320, 0.2481893805649713,  0.25043951279646320, 0.2481893805649713]
@timeit to "Basis" basis = jED.Basis(3)
ϵₖ = [0.3, 0.4]
Vₖ = [0.6, 0.7]
U  = 1.2
μ  = 0.6
β  = 13.4
@timeit to "Model" model = AIM(ϵₖ, Vₖ, μ, U)
@timeit to "Eigen problem" es = Eigenspace(model, basis);
Z = calc_Z(es, β)
E = calc_E(es, β)
println("E₀ = $(es.E0)\nZ  = $Z\nE  = $E")

freq_arr = [1im * (2*n+1)*π/β for n in 0:200]
@timeit to "GF" GF = calc_GF_1(es, freq_arr, β)
