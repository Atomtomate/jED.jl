using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using TimerOutputs
to = TimerOutput()



function DMFT_Loop(U::Float64, μ_in::Float64, β::Float64, dens_fix::Float64; maxit = 20)
    ϵₖ = [0.0235, 0.2177,-0.1389,-0.4502]#, 0.6248, -0.924] 
    Vₖ = [0.1496,  0.174,  0.201, 0.2841]#,  0.161, 0.2873]
    p  = AIMParams(ϵₖ, Vₖ)
    tsc = 0.40824829046386307/2
    Nν  = 1000
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
        println("Solution using Lsq: U8NS4   ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                    U8NS4   Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
        println("μ = $μ , dens = $dens")
        if i > 5
            μ = μ - (dens - dens_fix)/10
        end
    end
    return p, νnGrid, GImp_i, ΣImp_i, dens
end

#     U  = 0.2
#     μ  = U/2
#     β  = 4.0
# params_list = [(1.0, 0.5, 10),(1.0, 0.5, 11),(1.0, 0.5, 12),(1.0, 0.5, 13),(1.0, 0.5, 14),(1.0, 0.5, 15),(1.0, 0.5, 16),(1.0, 0.5, 17),
#                (0.75,0.375, 10),(0.75,0.375, 12),(0.75,0.375, 14),(0.75,0.375, 16),(0.75,0.375, 18),(0.75,0.375, 20),(0.75,0.375, 22),(0.75,0.375, 24),(0.75,0.375, 26),
#                 ]
U = 8.0
μ = 4.0
β = 55.0
dens = 1.0
p, νnGrid, GImp, ΣImp, dens = DMFT_Loop(U, μ, β, dens, maxit = 300);
