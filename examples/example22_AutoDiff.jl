using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using ForwardDiff
using LsqFit

NSites = 4
ϵₖ = [1.0, 0.5, -1.1, -0.6]
Vₖ = [0.25, 0.35, 0.45, 0.55]
p  = AIMParams(ϵₖ, Vₖ)
μ  = 0.6
U  = 1.2
β  = 4.0
tsc= 0.40824829046386307/2
Nν = 1000
Nk = 20
α  = 0.2
GImp_i = nothing
GImp_i_old = nothing

kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)
basis  = jED.Basis(length(Vₖ) + 1);
νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
N =   length(ϵₖ)

model  = AIM(ϵₖ, Vₖ, μ, U)
G0W    = GWeiss(νnGrid, μ, p)
es     = Eigenspace(model, basis);
println("     Calculating GImp")
GImp_i, dens = calc_GF_1(basis, es, νnGrid, β)
ΣImp_i = Σ_from_GImp(G0W, GImp_i)
GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)

function GW_fit_real(νnGrid::Vector, p::Vector)::Vector
    tmp = jED.GWeiss_real(νnGrid, μ, p[1:N], p[(N+1):end])
    return tmp
end
p0        = vcat(p.ϵₖ, p.Vₖ)
target = vcat(real(GLoc_i.parent), imag(GLoc_i.parent))
fit = curve_fit(GW_fit_real, νnGrid.parent, target, p0; autodiff=:forwarddiff)
println("Solution using Lsq:    ϵₖ = $(lpad.(round.(fit.param[1:NSites],digits=4),9)...)")
println("                       Vₖ = $(lpad.(round.(fit.param[NSites+1:end],digits=4),9)...)")
println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
