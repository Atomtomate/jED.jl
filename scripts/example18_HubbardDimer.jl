using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using TimerOutputs
to = TimerOutput()


t = 1.3
U = 3.3

tMatrix = [0.0 t
           t 0.0]
UMatrix = [U   0.0
           0.0 U]
μ  = U/2 #0.6
β  = 19.4
basis = jED.Basis(2)
model = Hubbard(tMatrix, UMatrix, μ)
es = Eigenspace(model, basis);

νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:2000], 0:2000)
GF, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=0.0, with_density=true)
GF_approx, dens_approx = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-8, with_density=true)

Z = calc_Z(es, β)
E = calc_E(es, β)
E₀ = U/2 - sqrt((U^2) + 16 * t^2)/2

println("E₀ = $(es.E0) vs  analytical solution: $E₀ \nZ  = $Z\nE  = $E\nn=$dens")

