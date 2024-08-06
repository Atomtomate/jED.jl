using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using NLsolve, StaticArrays

function DMFT_solve(NSites::Int, U::Float64, β::Float64, dens_in::Float64, p0)
    # ϵₖ = vcat(randn(trunc(Int,NSites/2)) .* U .+ U,  randn(trunc(Int,NSites/2)) .* U .- U,  repeat([0.2], NSites%2))
    # Vₖ = randn(NSites) ./ 100.0 .+ 0.17
    # μ = U/2
    # p0 = vcat(ϵₖ, Vₖ, μ)

    Nk  = 200
    Nν  = 2000
    kG     = jED.gen_kGrid("2Dsc-0.25-0.075-0.05",Nk) #"3Dsc-$tsc", Nk)
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    basis  = jED.Basis(NSites+1);

    function DMFT_map(p_in)
        p  = AIMParams(p_in[1:NSites], p_in[NSites+1:2*NSites])
        μ = p_in[end]
        overlap= Overlap(basis, create_op(basis, 1)) # optional
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis, verbose=false);
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-13, overlap=overlap, with_density=false)
        Nup = calc_Nup(es, β, basis, model.impuritySiteIndex)
        Ndo = calc_Ndo(es, β, basis, model.impuritySiteIndex)
        dens = Nup + Ndo
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        println("p_p = $p")
        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("p = $p")
        return vcat(p.ϵₖ, p.Vₖ), dens
    end

    function condition(u)
        p_out, n = DMFT_map(u)
        return vcat(p_out .- u[1:2*NSites], n - dens_in)
    end
    # prob = NonlinearProblem(condition, p0)
    sol = nlsolve(condition, p0)

    return sol
end



function DMFT_Loop(NSites::Int, U::Float64, μ_in::Float64, β::Float64, dens_fix::Float64; maxit = 20)
    ϵₖ = vcat(randn(trunc(Int,NSites/2)) .* U .+ U,  randn(trunc(Int,NSites/2)) .* U .- U,  repeat([0.2], NSites%2))
    Vₖ = randn(NSites) ./ 100.0 .+ 0.17
    p  = AIMParams(ϵₖ, Vₖ)
    tsc = 0.40824829046386307/2
    Nν  = 2000
    Nk  = 200
    α   = 0.4
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing

    kG     = jED.gen_kGrid("2Dsc-0.25-0.075-0.05",200) #"3Dsc-$tsc", Nk)
    basis  = jED.Basis(length(Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    μ = μ_in
        
    for i in 1:maxit
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis, verbose=false);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-13, overlap=overlap, with_density=false)
        Nup = calc_Nup(es, β, basis, model.impuritySiteIndex)
        Ndo = calc_Ndo(es, β, basis, model.impuritySiteIndex)
        dens = Nup + Ndo
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        !isnothing(GImp_i_old) && sum(abs.(GImp_i_old .- GImp_i)) < 1e-8 && break
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("Solution using Lsq:  U0  ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                     U0  Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
        println("it[$i]: μ = $μ , dens = $dens")
        if i > 5
            μ = μ - (dens - dens_fix)/10
        end
    end
    return p, νnGrid, GImp_i, ΣImp_i, μ, dens
end

U = 0.1
μ = 0.14
β = 55.0
dens = 1.0
NSites = 4
p_it, νnGrid, GImp, ΣImp, μ_it, dens = DMFT_Loop(NSites, U, μ, β, dens, maxit = 10);
p0 = vcat(p_it.ϵₖ, p_it.Vₖ, μ_it)
p_fix = DMFT_solve(NSites, U, β, dens, p0)
