# These tests are only here for stability reasons,
@testset "Observables" begin
    Z = calc_Z(es, β)
    E = calc_E(es, β)
    @test Z ≈ 1.020663095  atol=1e-8
    @test E ≈ 0.01047825 atol=1e-8
end
