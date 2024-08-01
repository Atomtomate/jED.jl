using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using TimerOutputs
to = TimerOutput()
using NLsolve
using FiniteDiff
using NLSolvers
using LsqFit
using Memoization

#jED.show(stdout,MIME("text/plain"),basis)

function DMFT_Loop(U::Float64, μ::Float64, β::Float64; maxit = 20)
    #ϵₖ = [189.2073, 64.9715,  -0.7088, 0.7055,   0.1413,  -0.1431]
    #Vₖ = [0.0005,  -0.0564, 0.274, 0.2744,  0.2192, 0.2196]
    ϵₖ = [10.0, 5.0, -3.0, 0.5]#, 0.1, 0.2]
    Vₖ = [0.25, 0.35, 0.45, 0.55]#, 0.6,0.65]
    p  = AIMParams(ϵₖ, Vₖ)
    tsc = 0.40824829046386307/2
    Nν  = 200
    Nk  = 80
    α   = 0.4
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing
    dens = NaN
    d = NaN
    Nup=NaN
    Ndo=NaN

    kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)
    basis  = jED.Basis(length(Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    N =   length(ϵₖ)

    for i in 1:maxit
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-16, overlap=overlap, with_density=true)
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
    end
    return p, νnGrid, GImp_i, ΣImp_i, dens
end

#     U  = 0.2
#     μ  = U/2
#     β  = 4.0
# params_list = [(1.0, 0.5, 10),(1.0, 0.5, 11),(1.0, 0.5, 12),(1.0, 0.5, 13),(1.0, 0.5, 14),(1.0, 0.5, 15),(1.0, 0.5, 16),(1.0, 0.5, 17),
#                (0.75,0.375, 10),(0.75,0.375, 12),(0.75,0.375, 14),(0.75,0.375, 16),(0.75,0.375, 18),(0.75,0.375, 20),(0.75,0.375, 22),(0.75,0.375, 24),(0.75,0.375, 26),
#                 ]
U = 0.01
μ = 0.005
β = 20.0
pp, νnGrid, GImp, ΣImp, dens = DMFT_Loop(U, μ, β, maxit = 30);

##################################################################








##################################################################
model  = AIM(pp.ϵₖ,pp.Vₖ, μ, U);
basis  = jED.Basis(length(pp.Vₖ) + 1);
es     = Eigenspace(model, basis);

d= jED.calc_D(es, β, basis, model.impuritySiteIndex)

Nup= jED.calc_Nup(es, β, basis, model.impuritySiteIndex)
Ndo= jED.calc_Ndo(es, β, basis, model.impuritySiteIndex)

println("d=$d    Nup=$Nup     Ndo=$Ndo")
