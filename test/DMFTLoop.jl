@testset "Greens Function Stuff" begin
    p = AIMParams([3.0, 4.0],[1.0,2.0])
    νnGrid = jED.FermionicMatsubaraGrid(1.0im .* collect(2:3), 0:1)
    Δi = jED.Δ_AIM(νnGrid, p)
    @test Δi[0] ≈ conj(1^2/(2im-3.0) + 2^2/(2im-4))
    @test Δi[1] ≈ conj(1^2/(3im-3.0) + 2^2/(3im-4))
end

@testset "DMFTLoop" begin
    ϵₖ = [0.1, 0.2]
    Vₖ = [0.2, 0.3]
    p  = AIMParams(ϵₖ, Vₖ)
    μ  = 0.6
    U  = 1.2
    β  = 13.4
    kG = jED.gen_kGrid("3Dsc-0.22", 40)

    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:1000], 0:1000)
    Δi     = Δ_AIM(νnGrid, p)
    GWeiss = GWeiss(νnGrid, μ, p)

    basis = jED.Basis(3)
    model = AIM(ϵₖ, Vₖ, μ, U)
    es = Eigenspace(model, basis);
    GImp_old = nothing
    maxit = 100

    for i in 1:maxit
        GImp_i = calc_GF_1(basis, es, νnGrid, β)
        ΣImp_i = Σ_from_GImp(GWeiss, GImp_i)
        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        GWeiss_i = GWeiss_from_Imp(GLoc_i, ΣImp_i)
        fit_AIM_params!(p, GWeiss_i, μ, νnGrid)
        if GImp_old != nothing && sum(abs.(GImp_old .- GImp_i)) < 1e-8
            break
        end
        GImp_old = deepcopy(GImp_i)
    end

end
