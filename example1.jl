using Pkg
Pkg.activate(@__DIR__)
using jED
using TimerOutputs
to = TimerOutput()


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
U  = 1.2
μ  = 0.6
β  = 13.4
@timeit to "Basis" basis = jED.Basis(length(ϵₖ)+1)
model = AIM(ϵₖ, Vₖ, μ, U)

# internally, the Hamiltonian is constructed with the follwing call.
# This will construct the full (!) Hamiltonian when called with the full basis (not segmentd into blocks)
# H = calc_Hamiltonian(model, basis)

@timeit to "Basis Full" es = Eigenspace(model, basis);
# @timeit to "Basis Lanczos" es2 = jED.Eigenspace_L(model, basis);

Z = calc_Z(es, β)
E = calc_E(es, β)
println("E₀ = $(es.E0)\nZ  = $Z\nE  = $E")

νnGrid = [1im * (2*n+1)*π/β for n in 0:10]
GF = calc_GF_1(basis, es, νnGrid, β)
GF_approx = calc_GF_1(basis, es, νnGrid, β, prefac_cut=1e-12)

