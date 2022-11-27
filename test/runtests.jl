using jED
using Test



@testset "states" begin
    include("states.jl")
end

@testset "Model" begin
    include("Models.jl")
end

@testset "Hamiltonian" begin
    include("Hamiltonian.jl")
end

@testset "jED.jl" begin

end
