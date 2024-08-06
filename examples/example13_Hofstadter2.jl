using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using Printf
using TimerOutputs

global to = TimerOutput()

function DMFT_Loop(U, μ, β, tsc, p_init; maxit = 20, Nν=300, Nk=40, α=0.4, GImp_cutoff=1e-12)
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing
    GLoc_i = nothing
    GLoc_i_old = nothing

    kG     = jED.gen_kGrid("Hofstadter:1:3-$tsc-0.025-0.002", Nk)
    basis  = jED.Basis(length(Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1))
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)

    for i in 1:maxit
        println("    ========== ITERATION $(rpad(i,3)) ==========    ")
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        @timeit to "GImp" GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=GImp_cutoff, overlap=overlap)
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        println("     Calculating GLoc")
        @timeit to "GLoc new" GLoc_i = jED.GLoc_MO(ΣImp_i, μ, νnGrid, kG)



        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
    end
    return p, νnGrid, GImp_i, GLoc_i, GLoc_i_old, ΣImp_i
end

ϵₖ = [-0.4, 0.4, -1.1, -0.6, 1.1, 0.6]
Vₖ = [0.25, 0.35, 0.45, 0.55, 0.50,0.45]
p_init  = AIMParams(ϵₖ, Vₖ)
p_step1, νnGrid, GImp_step1, GLoc_step1, GLoc_old_step1, ΣImp_step1 = DMFT_Loop(1.2, 0.6, 40.0, 0.25, p_init, maxit = 30, Nν=50, GImp_cutoff=1e-6)
p_step2, νnGrid, GImp_step2, GLoc_step2, GLoc_old_step2, ΣImp_step2 = DMFT_Loop(1.2, 0.6, 40.0, 0.25, p_step1, maxit = 10, Nν=500, GImp_cutoff=1e-12)


# open("gm_wim", "w") do io
#     res = map(x -> rpad.(@sprintf("%0.12f",x), 20), [imag(νnGrid) real(GImp) imag(GImp)])
#     writedlm(io, res)
# end

