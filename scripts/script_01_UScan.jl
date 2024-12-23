using Pidfile
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED

nsites = parse(Int, ARGS[1])
from0  = parse(Bool, ARGS[2])
β      = parse(Float64, ARGS[3])
out_path  = ARGS[4]
ID     = parse(Int, ARGS[5])

ek_start = if nsites == 4 
        [0.0298, 0.2937,-0.1117,-0.5875] 
    elseif nsites == 5 
        [0.0291,  0.2296, -0.1289, -0.4249,  0.6445]
    elseif nsites == 6
        [0.0291,  0.2296, -0.1289, -0.4249,  0.6445,  -0.887]
    end
vk_start = if nsites == 4 
        [0.1384, 0.2088, 0.2032, 0.3954] 
    elseif nsites == 5 
        [0.153, 0.1794,  0.1985,  0.2763,  0.1635]
    elseif nsites == 6
        [0.153, 0.1794,  0.1985,  0.2763,  0.1635,  0.2902]
end

function DMFT_Loop(U::Float64, μ_in::Float64, β::Float64, dens_fix::Float64, p; maxit = 20)
    Nν  = 2000
    Nk  = 200
    α   = 0.4
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing

    kG     = jED.gen_kGrid("2Dsc-0.25-0.075-0.05",Nk)
    basis  = jED.Basis(length(p.Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    μ = μ_in
    dens = NaN
    model  = nothing
    es = nothing
        
    for i in 1:maxit
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis, verbose=false);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-14, overlap=overlap, with_density=false)
        Nup = calc_Nup(es, β, basis, model.impuritySiteIndex)
        Ndo = calc_Ndo(es, β, basis, model.impuritySiteIndex)
        dens = Nup + Ndo
        (i < 10) && !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        !isnothing(GImp_i_old) && sum(abs.(GImp_i_old .- GImp_i)) < 1e-8 && break
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("Solution using Lsq:   ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                      Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
        println("μ = $μ , dens = $dens")
        if i > 5
            μ = μ - (dens - dens_fix)/10
        end
    end
    D   = jED.calc_D(es, β, basis, model.impuritySiteIndex)
    return p, νnGrid, GImp_i, ΣImp_i,μ, dens, D
end

function run_scan()
    p = AIMParams(ek_start, vk_start)
    Ur = from0 ? LinRange(0,6,51) : reverse(LinRange(0,6,51))
    # Ur = from0 ? LinRange(0,6,26)[7:20] : reverse(LinRange(0,6,26)[7:20])
    μ = first(Ur)/2 
    for Ui in Ur
        p, νnGrid, GImp, ΣImp, μ, dens, D = DMFT_Loop(Ui, μ, β, 1.0, p, maxit = 400);
        mkpidlock(joinpath(out_path,"lock_$ID.pid")) do
            open(joinpath(out_path,"results_$ID.txt"), lock=true, append=true) do f
                println(f, "$Ui,$β,$μ,$dens,$D,$(p.ϵₖ),$(p.Vₖ)")
            end
        end
    end
end
run_scan()
