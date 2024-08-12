# These tests are only here for stability reasons,
@testset "Observables" begin
    Z = calc_Z(es, β)
    E = calc_E(es, β)
    EKin = calc_EKin_DMFT(νnGrid, ϵₖ, Vₖ, GImp, nden, U, β, μ)
    EPot = calc_EPot_DMFT(νnGrid, ϵₖ, Vₖ, GImp, nden, U, β, μ)
    @test Z ≈ 1.020663095  atol=1e-8
    @test E ≈ 0.01047825 atol=1e-8
    #This is only true for the DMFT solution: @test E ≈ EKin + EPot
end
