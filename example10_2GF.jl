using Pkg
Pkg.activate(@__DIR__)
using jED
using TimerOutputs
to = TimerOutput()

ϵₖ = [
    0.2 ,
    1.1]
Vₖ = [
      0.3,
      0.4]
U  = 1.1
μ  = 0.5
β  = 3.0
NB = length(ϵₖ) 
@timeit to "Basis" basis = jED.Basis(NB+1)
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
GF, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=0.0)
GF_approx, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-8)
state = 1
op_up =  create_op(basis, 1)
overlap_up =  Overlap(basis, op_up)
op_do =  create_op(basis, (NB+1)+1)
overlap_do =  Overlap(basis, op_do)
r1, r2, r3, r4, r5 = jED.lehmann_full(basis, es, overlap_up, β, 0.0)
r1_do, r2_do, r3_do, r4_do, r5_do = jED.lehmann_full(basis, es, overlap_do, β, 0.0)
