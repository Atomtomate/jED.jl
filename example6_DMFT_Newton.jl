using Pkg
Pkg.activate(@__DIR__)
using jED
using NLsolve
using FiniteDiff
using NLSolvers
using LsqFit

ϵₖ = [1.0, 0.5, -1.1, -0.6]
Vₖ = [0.25, 0.35, 0.45, 0.55]
p  = AIMParams(ϵₖ, Vₖ)
μ  = 0.6
U  = 1.2
β  = 4.0
tsc= -0.40824829046386307/2
Nν = 1000
Nk = 20
α  = 0.2
GImp_i = nothing
GImp_i_old = nothing

kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)
basis  = jED.Basis(length(Vₖ) + 1);
νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
N =   length(ϵₖ)

function DMFT_solve_test(p_vec::Vector)
    p      = AIMParams(p_vec)
    model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
    G0W    = GWeiss(νnGrid, μ, p)
    es     = Eigenspace(model, basis);
    isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
    println("     Calculating GImp")
    GImp_i = calc_GF_1(basis, es, νnGrid, β)
    ΣImp_i = Σ_from_GImp(G0W, GImp_i)
    return ΣImp_i
end


