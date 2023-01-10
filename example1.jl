using Pkg
Pkg.activate(@__DIR__)
using jED

# @timeit to "Basis" basis = jED.Basis(7)
# ϵₖ = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# Vₖ = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
# basis = jED.Basis(5)
#ϵₖ = [0.24379126936809078, 1.1042333108436919, -0.24379126936809078, -1.1042333108436919]
#Vₖ = [0.25917560591147798, 0.23335139853990977, 0.25917560591147798,  0.23335139853990977]
#ϵₖ = [0.23720747964147854, 1.0761589467007526, -0.23720747964147854, -1.076158946700752]
#Vₖ = [0.25043951279646320, 0.2481893805649713,  0.25043951279646320, 0.2481893805649713]
basis = jED.Basis(3)
# ϵₖ = [ -0.27609287995684029, 0.27618973571133110]
# Vₖ = [0.29192047882034178, 0.29192904949573512]

ϵₖ = [
 -0.27609253559340058     ,
    0.27618797819209367     ]
Vₖ = [
  0.29191987451883261     ,
    0.29192835679140872     ]
U  = 1.2
μ  = 0.6
β  = 13.4
model = AIM(ϵₖ, Vₖ, μ, U)

# internally, the Hamiltonian is constructed with the follwing call.
# This will construct the full (!) Hamiltonian when called with the full basis (not segmentd into blocks)
# H = calc_Hamiltonian(model, basis)

es = Eigenspace(model, basis);

Z = calc_Z(es, β)
E = calc_E(es, β)
println("E₀ = $(es.E0)\nZ  = $Z\nE  = $E")

freq_arr = [1im * (2*n+1)*π/β for n in 0:10]
#GF = calc_GF_1(es, basis, freq_arr, β)
