using Pkg
Pkg.activate(@__DIR__)
using jED
using Printf
using TimerOutputs

global to = TimerOutput()

function DMFT_Loop(;maxit = 30)
    ϵₖ = [1.859012640410684,
-1.234144804393995,
-0.327074265095705,
0.164928397400499]
    Vₖ = [0.21351085528763,
0.299231813327473,
0.269996433605904,
0.203793698791301]
    p  = AIMParams(ϵₖ, Vₖ)
    μ  = 0.08 #0.4823938
    U  = 2.0
    β  = 1.0
    Nν = 500
    Nk = 60
    α  = 0.7
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing
    GLoc_i = nothing
    GLoc_i_old = nothing

    kG     = jED.gen_kGrid("fcc-0.14433756729740646-0.0-0.0", Nk)
    #kG     = jED.gen_kGrid("3dsc-0.20412414523193154", Nk)
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
        @timeit to "GImp" GImp_i = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-12, overlap=overlap, print_density=true)
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        println("     Calculating GLoc")
        #@timeit to "GLoc" GLoc_i_old = jED.GLoc_MO_old(ΣImp_i, μ, νnGrid, kG)
        #@timeit to "GLoc old2" GLoc_i_old = jED.GLoc_MO_old2(ΣImp_i, μ, νnGrid, kG)
        @timeit to "GLoc new" GLoc_i = jED.GLoc(ΣImp_i, μ, νnGrid, kG)



        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
    end
    return p, νnGrid, GImp_i, GLoc_i, GLoc_i_old, ΣImp_i
end

p, νnGrid, GImp_res, GLoc_res, GLoc_old_res, ΣImp = DMFT_Loop(maxit = 30)

# open("gm_wim", "w") do io
#     res = map(x -> rpad.(@sprintf("%0.12f",x), 20), [imag(νnGrid) real(GImp) imag(GImp)])
#     writedlm(io, res)
# end
