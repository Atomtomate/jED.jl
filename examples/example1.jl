using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED

# jED.show(stdout,MIME("text/plain"), VARIABLE)



# ϵₖ = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# Vₖ = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
# basis = jED.Basis(5)
#ϵₖ = [0.24379126936809078, 1.1042333108436919, -0.24379126936809078, -1.1042333108436919]
#Vₖ = [0.25917560591147798, 0.23335139853990977, 0.25917560591147798,  0.23335139853990977]
#ϵₖ = [0.23720747964147854, 1.0761589467007526, -0.23720747964147854, -1.076158946700752]
#Vₖ = [0.25043951279646320, 0.2481893805649713,  0.25043951279646320, 0.2481893805649713]
#basis = jED.Basis(3)
# ϵₖ = [ -0.27609287995684029, 0.27618973571133110]
# Vₖ = [0.29192047882034178, 0.29192904949573512]


ϵₖ = [
    1.0 ,
    0.5 ,
    -1.1,
    -0.6]
Vₖ = [
      0.250,
      0.350,
      0.450,
      0.550]
U  = 3.2
μ  = U/2 #0.6
β  = 13.4
basis = jED.Basis(length(ϵₖ)+1)
model = AIM(ϵₖ, Vₖ, μ, U)

# internally, the Hamiltonian is constructed with the follwing call.
# This will construct the full (!) Hamiltonian when called with the full basis (not segmentd into blocks)
# H = calc_Hamiltonian(model, basis)

es = Eigenspace(model, basis);
# @timeit to "Basis Lanczos" es2 = jED.Eigenspace_L(model, basis);

νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:2000], 0:2000)
GF, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=0.0, with_density=true)
GF_approx, dens_approx = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-8, with_density=true)

Z = calc_Z(es, β)
E = calc_E(es, β)
println("E₀ = $(es.E0)\nZ  = $Z\nE  = $E\nn=$dens")

