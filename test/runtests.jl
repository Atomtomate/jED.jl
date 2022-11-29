using jED
using Test


basis = jED.Basis(3)
s_test = basis.states[63:-1:61];
ϵₖ = [0.3, 0.8]
Vₖ = [0.6, 0.7]
U  = 1.0
μ  = 0.5
β  = 10.0
model = AIM(ϵₖ, Vₖ, μ, U)
H_test = jED.calc_Hamiltonian(model, s_test)
es = Eigenspace(model,basis)

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

@testset "jED.jl" begin

end
