using jED
using Test
using LinearAlgebra


basis = jED.Basis(3)
s_test = basis.states[63:-1:61];
ϵₖ = [0.3, 0.8]
Vₖ = [0.6, 0.7]
U  = 1.0
μ  = 0.5
β  = 10.0
model = AIM(ϵₖ, Vₖ, μ, U)
H_test = jED.calc_Hamiltonian(model, s_test)
es = Eigenspace(model,basis,verbose=false)
νnGrid = [(2*n+1)*π/β for n in 0:1500]
GImp,nden = calc_GF_1(basis, es, νnGrid, β; with_density=true)

@testset "states" begin
    include("States.jl")
end

@testset "Operators" begin
    include("Operators.jl")
end

@testset "Model" begin
    include("Models.jl")
end

@testset "Eigenspace" begin
    include("Eigenspace.jl")
end

@testset "Observables" begin
    include("Observables.jl")
end

@testset "Greens Functions" begin
    include("GreensFunctions.jl")
end

@testset "DMFTLoop" begin
    include("DMFTLoop.jl")
end

@testset "jED.jl" begin
    include("full_N2_run.jl")
end
