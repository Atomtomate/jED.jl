using Pkg
Pkg.activate(@__DIR__)
using jED
using TimerOutputs
to = TimerOutput()



function DMFT_Loop(;maxit = 20)
    ϵₖ = [1.0, 0.5, -1.1, -0.6]
    Vₖ = [0.25, 0.35, 0.45, 0.55]
    p  = AIMParams(ϵₖ, Vₖ)
    μ  = 0.6
    U  = 1.2
    β  = 4.0
    tsc = -0.40824829046386307/2
    Nν  = 1000
    Nk  = 20
    α   = 0.2
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing

    kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)
    basis  = jED.Basis(length(Vₖ) + 1);
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
        
    for i in 1:maxit
        println(" ==== Iteration $i ====")
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        GImp_i = calc_GF_1(basis, es, νnGrid, β)
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("     iteration $i with AIM params ∑ Vₖ^2 = $(sum(p.Vₖ .^ 2)), checksum GImp = $(abs(sum(GImp_i)))")
        display(p)
    end
    return p, νnGrid, GImp_i, ΣImp_i
end

p, νnGrid, GImp, ΣImp = DMFT_Loop(maxit = 30)
