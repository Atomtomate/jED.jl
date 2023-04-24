using Pkg
Pkg.activate(@__DIR__)
using jED
using Printf
using TimerOutputs

global to = TimerOutput()

function DMFT_Loop(;maxit = 20)
    ϵₖ = [1.0, 0.5, -1.1, -0.6, 0.7]
    Vₖ = [0.25, 0.35, 0.45, 0.55, 0.60]
    p  = AIMParams(ϵₖ, Vₖ)
    μ  = 0.6
    U  = 1.2
    β  = 40.0
    tsc= 0.25
    Nν = 1000
    Nk = 40
    α  = 0.4
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing
    GLoc_i = nothing
    GLoc_i_old = nothing

    kG     = jED.gen_kGrid("Hofstadter:5:17-$tsc", Nk)
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
        @timeit to "GImp" GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-12, overlap=overlap)
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        println("     Calculating GLoc")
        #@timeit to "GLoc" GLoc_i_old = jED.GLoc_MO_old(ΣImp_i, μ, νnGrid, kG)
        #@timeit to "GLoc old2" GLoc_i_old = jED.GLoc_MO_old2(ΣImp_i, μ, νnGrid, kG)
        @timeit to "GLoc new" GLoc_i = jED.GLoc_MO(ΣImp_i, μ, νnGrid, kG)



        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
    end
    return p, νnGrid, GImp_i, GLoc_i, GLoc_i_old, ΣImp_i
end

p, νnGrid, GImp, GLoc, GLoc_old, ΣImp = DMFT_Loop(maxit = 2)

# open("gm_wim", "w") do io
#     res = map(x -> rpad.(@sprintf("%0.12f",x), 20), [imag(νnGrid) real(GImp) imag(GImp)])
#     writedlm(io, res)
# end
