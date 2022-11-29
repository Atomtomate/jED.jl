@testset "calc_Hamiltonian" begin
    @test all(H_test .≈ [1.7 0.6 -0.7;0.6 1.9 0.0;-0.7 0.0 1.4]) # carefully calculated by hand
    @test es.E0 ≈ -1.6353165 atol=1e-6
end
