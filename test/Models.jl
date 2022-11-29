@testset "AIM" begin
    ϵₖ = [0.5, -5.0]
    ϵₖ2 = [0.5, -5.0, 0.0]
    Vₖ = [1.0, 1.0]
    Vₖ2 = [1.0 + 1im, 1.0]
    U  = 1.0
    μ  = 0.5
    model = AIM(ϵₖ, Vₖ, μ, U)
    model2 = AIM(ϵₖ, Vₖ2, μ, U)
    @test_throws ArgumentError AIM(ϵₖ2, Vₖ, μ, U)
    @test all(model.tMatrix .≈ [-μ Vₖ[1] Vₖ[2]; Vₖ[1] ϵₖ[1] 0; Vₖ[2] 0 ϵₖ[2]])
    @test all(model.JMatrix .≈ zeros(3,3))
    @test model.UMatrix[1,1] ≈ U
    @test all(model2.tMatrix .≈ model2.tMatrix')
end
