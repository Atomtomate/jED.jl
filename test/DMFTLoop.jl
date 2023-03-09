@testset "Greens Function Stuff" begin
    p = AIMParams([3.0, 4.0],[1.0,2.0])
    νnGrid = jED.FermionicMatsubaraGrid(1.0im .* collect(2:3), 0:1)
    Δi = jED.Δ_AIM(νnGrid, p)
    @test Δi[0] ≈ conj(1^2/(2im-3.0) + 2^2/(2im-4))
    @test Δi[1] ≈ conj(1^2/(3im-3.0) + 2^2/(3im-4))
end

@testset "DMFTLoop" begin

    ϵₖ = [1.0, 0.5, -1.1, -0.6]
    Vₖ = [0.25, 0.35, 0.45, 0.55]
    p  = AIMParams(ϵₖ, Vₖ)
    μ  = 0.6
    U  = 1.2
    β  = 13.4
    tsc = -0.40824829046386307/2
    Nν  = 10
    Nk  = 2

    kG = jED.gen_kGrid("3Dsc-$tsc", Nk)

    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    G0W    = GWeiss(νnGrid, μ, p)
    basis = jED.Basis(5);
        
    GImp_i = nothing
    maxit = 3

    for i in 1:maxit
        model  = AIM(ϵₖ, Vₖ, μ, U)
        es     = Eigenspace(model, basis);
        GImp_i = calc_GF_1(basis, es, νnGrid, β)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        G0W    = GWeiss_from_Imp(GLoc_i, ΣImp_i)
        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
    end

end
