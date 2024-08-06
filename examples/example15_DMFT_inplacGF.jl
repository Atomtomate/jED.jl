# This example demonstrates the use of the inplace calculation of the 1-particle GF, which improves performance slightly, for very small cutoffs for energy differences in the GF calculatation.. 
# The inplace version does not store the overval and energy differences and is therefore more difficult to extenThe inplace version does not store the overval and energy differences and is therefore more difficult to extend.
# Consider starting julia with `julia --check-bounds=no --inline=yes -O3` options in performance critical runs.
# If startup time is critical run the code once with --output-ji jED.ji and later include -J jED.ji in each start (this will save the precompiled state of the program and use it in subsequent runs)

using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using TimerOutputs
to = TimerOutput()



function DMFT_Loop(U::Float64, μ::Float64, β::Float64; maxit = 20)
    ϵₖ = [1.0, 0.5, -1.0, -0.5]#, 1.0, -1.0]
    Vₖ = [0.25, 0.35, 0.25, 0.35]#, 0.6, 0.7]
    p  = AIMParams(ϵₖ, Vₖ)
    tsc = 0.40824829046386307/2
    Nν  = 500
    Nk  = 40
    α   = 0.4
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing

    kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)
    basis  = jED.Basis(length(Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
        
    for i in 1:maxit
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        @timeit to "GF calc" GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=0.0, overlap=overlap)
        println("     Calculating GImp (inplace)")
        @timeit to "GF inplace calc" GImp_i2, dens2 = calc_GF_1_inplace(basis, es, νnGrid, β, overlap, 0.0)
        println(all(GImp_i.parent .≈ GImp_i2))
        println(dens ≈ dens2)
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

#     U  = 0.2
#     μ  = U/2
#     β  = 4.0
# params_list = [(1.0, 0.5, 10),(1.0, 0.5, 11),(1.0, 0.5, 12),(1.0, 0.5, 13),(1.0, 0.5, 14),(1.0, 0.5, 15),(1.0, 0.5, 16),(1.0, 0.5, 17),
#                (0.75,0.375, 10),(0.75,0.375, 12),(0.75,0.375, 14),(0.75,0.375, 16),(0.75,0.375, 18),(0.75,0.375, 20),(0.75,0.375, 22),(0.75,0.375, 24),(0.75,0.375, 26),
#                 ]
p, νnGrid, GImp, ΣImp = DMFT_Loop(2.0, 1.0, 15.0, maxit = 4)
println(to)
