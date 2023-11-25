using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using TimerOutputs
to = TimerOutput()

const FPT = jED.Float64x2


function DMFT_Loop(U::Float64, μ::Float64, β::Float64; maxit = 20)
    ϵₖ = [10.0, 5.0, -3.0, 0.5]
    Vₖ = FPT[0.25, 0.35, 0.45, 0.55]
    p  = AIMParams(ϵₖ, Vₖ)
    tsc = 0.40824829046386307/2
    Nν  = 1000
    Nk  = 40
    α   = 0.4
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing

    kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)
    basis  = jED.Basis(length(Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector{Complex{FPT}}([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
        
    for i in 1:maxit
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-16, overlap=overlap)
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
    end
    return p, νnGrid, GImp_i, ΣImp_i
end

#                 ]
p, νnGrid, GImp, ΣImp = DMFT_Loop(2.0, 1.0, 15.0, maxit = 30)

