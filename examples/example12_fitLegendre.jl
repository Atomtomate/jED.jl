using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using IRBasis


ϵₖ = [1.0, 0.5, -1.1, -0.6]     # Anderson bath level parameters
Vₖ = [0.25, 0.35, 0.45, 0.55]   # Anderson hybridization parameters
p  = AIMParams(ϵₖ, Vₖ)          # Anderson Parameters (struct type)
μ  = 0.6                        # chemical potentian
U  = 1.2                        # Coulomb interaction
β  = 4.0                        # inverse temperature
tsc= 0.40824829046386307/2      # hopping amplitude
Nν = 1000                       # Number of fermionic frequencies
Nk = 40                         # Number of k-sampling points i neach direction
ωcut = 10.0                     # Energy cutoff for IR Bases

# Grids, Bath and IRBasis
kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)                 # struct for k-grid
νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1) # Fermionic frequency grid
basis  = jED.Basis(length(Vₖ)+1);                       # ED basis
imp_OP = create_op(basis, 1)                            # Operator for impurity measurement
overlap= Overlap(basis, imp_OP)                         # helper for <i|OP|j> overlaps for 
G0W    = GWeiss(νnGrid, μ, p)                           # Bath Green's function
# basisIR = FiniteTempBasis{SparseIR.Fermionic}(β, ωcut)  # IR Basis for minimal GF representation
# sτ  = TauSampling(basis)
# siω = MatsubaraSampling(basis; positive_only=true)

# Impurity solution
model  = AIM(ϵₖ, Vₖ, μ, U)
es     = Eigenspace(model, basis);
GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-12, overlap=overlap, with_density=true)
ΣImp_i = Σ_from_GImp(G0W, GImp_i)

# Fits
fit_AIM_params!(p, GLoc_i, μ, νnGrid)

#TODO: finish this
