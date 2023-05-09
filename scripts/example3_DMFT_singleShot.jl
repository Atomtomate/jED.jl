using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED


ϵₖ = [1.0, 0.5, -1.1, -0.6]
Vₖ = [0.25, 0.35, 0.45, 0.55]
p  = AIMParams(ϵₖ, Vₖ)
μ  = 0.6
U  = 1.2
β  = 4.0
tsc= 0.40824829046386307/2
Nν = 1000
Nk = 20

kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)
basis  = jED.Basis(5);
νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
G0W    = GWeiss(νnGrid, μ, p)

model  = AIM(ϵₖ, Vₖ, μ, U)
es     = Eigenspace(model, basis);
GImp_i, dens = calc_GF_1(basis, es, νnGrid, β)
ΣImp_i = Σ_from_GImp(G0W, GImp_i)

GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
G0W    = GWeiss_from_Imp(GLoc_i, ΣImp_i)
fit_AIM_params!(p, GLoc_i, μ, νnGrid)
