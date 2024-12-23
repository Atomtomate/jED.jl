using Pidfile
# using Pkg
# Pkg.activate(joinpath(@__DIR__,".."))
using jED
using LsqFit

U     = parse(Float64, ARGS[1])
β     = parse(Float64, ARGS[2])
NSites= parse(Int, ARGS[3])
μ     = parse(Float64, ARGS[4])
KGStr = ARGS[5]
out_path  = ARGS[6]
ID  = ARGS[7]

function DMFT_Loop(U::Float64, μ::Float64, β::Float64, NSites::Int, KGStr::String; maxit = 20)
    ϵₖ =  randn(NSites)
    Vₖ = randn(NSites)
    p  = AIMParams(ϵₖ, Vₖ)
    Nν  = 1000
    Nk  = 100
    α   = 0.4
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing
    dens = NaN
    d = NaN
    Nup=NaN
    Ndo=NaN

    kG     = jED.gen_kGrid(KGStr, Nk)
    basis  = jED.Basis(length(Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    N =   length(ϵₖ)
    es = nothing
    model = nothing

    for i in 1:maxit
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis, verbose=false);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        #println("     Calculating GImp")
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-14, overlap=overlap, with_density=false)
        !isnothing(GImp_i_old) && sum(abs.(GImp_i_old .- GImp_i)) < 1e-8 && break
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        fit_AIM_params_Conj_Grad!(p, GLoc_i, μ, νnGrid)
        #println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        #println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        #println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
    end
    D   = jED.calc_D(es, β, basis, model.impuritySiteIndex)
    Nup = jED.calc_Nup(es, β, basis, model.impuritySiteIndex)
    Ndo = jED.calc_Ndo(es, β, basis, model.impuritySiteIndex)
    println("DONE: U=$U, μ=$μ, β=$β")
    return p, D, Nup, Ndo
end

params,D,Nup,Ndo = DMFT_Loop(U, μ, β, NSites, KGStr, maxit = 200);


mkpidlock(joinpath(out_path,"flock_$ID.pid")) do
    open(joinpath(out_path,"results_$ID.txt"), lock=true, append=true) do f
        println(f, "$U,$μ,$β,$D,$Nup,$Ndo,$(params.ϵₖ),$(params.Vₖ)")
    end
end
