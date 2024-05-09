using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using TimerOutputs
to = TimerOutput()

β  = 19.4

t = 2.0
U = 10.0

tMatrix = [0.0 -t
           -t 0.0]
UMatrix = [U   0.0
           0.0 U]
basis = jED.Basis(2, N_filter=[2], S_filter=[0])
model = Hubbard(tMatrix, UMatrix)
es = Eigenspace(model, basis);

#νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:2000], 0:2000)
#GF, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=0.0, with_density=true)
#GF_approx, dens_approx = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-8, with_density=true)

Z = calc_Z(es, β)
E = calc_E(es, β)
E₀ = U/2 - sqrt((U^2) + 16 * t^2)/2
E₁ = U/2 + sqrt((U^2) + 16 * t^2)/2

println("E₀ = $(es.E0) vs  analytic solution: $E₀ / $E₁ \nZ  = $Z\nE  = $E")
println("Input Hamiltonian")
calc_Hamiltonian(model, basis)

println("Diagonalized Hamiltonian")
show_diag_Hamiltonian(basis, es)


println("Energies (sorted in blocks)")
show_energies_states(basis, es)