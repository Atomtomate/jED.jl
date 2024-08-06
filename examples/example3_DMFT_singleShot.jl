using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED


ϵₖ = [
  0.424975691962297     ,
  0.994344725468974     ,
 -0.417200973730032     ,
 -0.995832136939354     
     ]
Vₖ = [
  0.281229779570778     ,
  0.202118597850295     ,
  0.243829596221534     ,
  0.192700102801193     
     ]
p  = AIMParams(ϵₖ, Vₖ)
μ  = 1.15
U  = 2.0
β  = 10.0
tsc= 0.25#0.40824829046386307/2
Nν = 1000
Nk = 100

kG     = jED.gen_kGrid("2Dsc-$tsc-0.025-0.002", Nk)
basis  = jED.Basis(5);
νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
G0W    = GWeiss(νnGrid, μ, p)

model  = AIM(ϵₖ, Vₖ, μ, U)
es     = Eigenspace(model, basis);
GImp_i, dens = calc_GF_1(basis, es, νnGrid, β)
ΣImp_i = Σ_from_GImp(G0W, GImp_i)

GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
G0W_i  = GWeiss_from_Imp(GLoc_i, ΣImp_i)
fit_AIM_params!(p, GLoc_i, μ, νnGrid)
