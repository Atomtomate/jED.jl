using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(joinpath(@__DIR__,".."))
@everywhere using jED
@everywhere using SparseIR
using HDF5

VkSamples = 2
EkSamples = 2
MuSamples = 2
betaSamples = 1
USamples = 1

NBath = 2
NCoeffs::Int = 10

βList = [30.0] #1 ./ LinRange(0.025,1,HubbParSamples)
UList = [1.0]

println("Total length: ", USamples*betaSamples*MuSamples*VkSamples^NBath*EkSamples^NBath)

Ui = UList[1]
V1 = LinRange(0,1.0,VkSamples)
E1 = LinRange(-2Ui, 2Ui, EkSamples)
μList = LinRange(-2*Ui, 2*Ui, MuSamples) 

fullParamList = collect(Base.product(E1,E1,V1,V1,μList,UList,βList))[:]
NSamples = length(fullParamList)
println("check: ", NSamples)


@everywhere function solve_imp(parList)
    # ========== TODO: this is only fixed for this test ============
    Nν::Int = 500
    NBath::Int = 2
    β::Float64 = 30.0
    U::Float64 = 1.0
    NCoeffs::Int = 10
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    νnGridL = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:10000-1], 0:10000-1)
    # ========== TODO: this is only fixed for this test ============

    # SparseIR Setup:
        function eval_sparseIR(siν, arr::jED.OffsetVector)
            map(x->arr[x.n-1], siν.sampling_points)
        end

        # Basis and Grid
        ωmax = 2.0*U
        ε = 1e-12
        IRbasis = FiniteTempBasis{Fermionic}(β, ωmax, ε, max_size=NCoeffs)
        siν = MatsubaraSampling(IRbasis; positive_only=true)
        νnGrid_IR = eval_sparseIR(siν, νnGrid)

        #Preallocs (use fit() instead of fit!() to avoid this)
        Gl      = Vector{ComplexF64}(undef, size(siν.matrix,2))
        #GF_test = Vector{ComplexF64}(undef, size(siν.matrix,1))
        workarr = Vector{ComplexF64}(undef, SparseIR.workarrlength(siν, νnGrid_IR))

        #Gl = fit(siν, GF_IR_pre; dim = 1);
        #GF_test = evaluate(siν, Gl)


        IRbasisΣ = AugmentedBasis(IRbasis, MatsubaraConst)
        siνΣ = MatsubaraSampling(IRbasisΣ; positive_only=true)
        νnGridΣ_IR = eval_sparseIR(siνΣ, νnGrid)

        Σl       = Vector{ComplexF64}(undef, size(siνΣ.matrix,2))
        #Σ_test   = Vector{ComplexF64}(undef, size(siνΣ.matrix,1))
        workarrΣ = Vector{ComplexF64}(undef, SparseIR.workarrlength(siνΣ, νnGridΣ_IR))

    # End SparseIR Setup

    basis  = jED.Basis(NBath+1);       # ED basis
    imp_OP = create_op(basis, 1)    # Operator for impurity measurement
    overlap= Overlap(basis, imp_OP)       # helper for <i|OP|j> overlaps for 
    # Σ0 = similar(νnGrid); fill!(Σ0, 0.0)
    #GBath = GLoc(Σ0, μ, νnGrid, kG)


    G0WMatrix   = Matrix{ComplexF64}(undef, Nν, length(parList))
    GImpMatrix = Matrix{ComplexF64}(undef, Nν, length(parList))
    ΣImpMatrix = Matrix{ComplexF64}(undef, Nν, length(parList))
    GLMatrix   = Matrix{ComplexF64}(undef, NCoeffs, length(parList))
    SLMatrix   = Matrix{ComplexF64}(undef, NCoeffs+1, length(parList))
    densList   = Vector{Float64}(undef, length(parList))

    for (it, el) in enumerate(parList)
        e1, e2, v1, v2, μ, U, β = el
        ϵₖ = [e1, e2]
        Vₖ = [v1, v2]

        p    = AIMParams(ϵₖ, Vₖ)          # Anderson Parameters (struct type)
        G0W  = GWeiss(νnGrid, μ, p)

        # Impurity solution
        model  = AIM(ϵₖ, Vₖ, μ, U)
        es     = Eigenspace(model, basis, verbose=false);
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-14, overlap=overlap, with_density=true)
       
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)

        GF_IR_pre = eval_sparseIR(siν, GImp_i)
        fit!(Gl, siν, GF_IR_pre; workarr=workarr);
        #evaluate!(GF_test, siν, Gl)

        Σ_IR_pre = eval_sparseIR(siνΣ, ΣImp_i)
        fit!(Σl, siνΣ, Σ_IR_pre; workarr=workarrΣ);
        #evaluate!(Σ_test, siνΣ, Σl);

        G0WMatrix[:,it] = G0W
        GImpMatrix[:,it] = GImp_i
        ΣImpMatrix[:,it] = ΣImp_i
        GLMatrix[:,it] = Gl
        SLMatrix[:,it] = Σl
        densList[it] = dens
    end
    return hcat(collect.(parList)...), G0WMatrix, GImpMatrix, ΣImpMatrix, GLMatrix, SLMatrix, densList
end
solve_imp(fullParamList[1:10])
NWorkers = length(workers())
ChunkSize = ceil(Int, NSamples/NWorkers)
batch_indices = collect(Base.Iterators.partition(1:length(fullParamList),ChunkSize))
futures = []
for (i,wi) in enumerate(workers())
    push!(futures, remotecall(solve_imp, wi, fullParamList[batch_indices[i]]))
end


Nν::Int = 500 
NParams = length(fullParamList[1])
paramsList_check = Array{Float64,2}(undef, NParams, NSamples)
G0WList = Array{ComplexF64,2}(undef, Nν, NSamples)
GImpList = Array{ComplexF64,2}(undef, Nν, NSamples)
ΣImpList = Array{ComplexF64,2}(undef, Nν, NSamples)
GLList = Array{ComplexF64,2}(undef, NCoeffs, NSamples)
SLList = Array{ComplexF64,2}(undef, NCoeffs+1, NSamples)
densList = Array{Float64,1}(undef, NSamples)
fill!(densList, NaN)

for (i,res) in enumerate(workers())
    res = fetch(futures[i])
    ind = batch_indices[i]
    paramsList_check[:, ind] = res[1] 
    G0WList[:, ind] = res[2]
    GImpList[:, ind] = res[3] 
    ΣImpList[:, ind] = res[4] 
    GLList[:, ind] = res[5] 
    SLList[:, ind] = res[6] 
    densList[ind] = res[7] 
end

fn = "test.hdf5"
h5open(fn, "w") do f
    gr = create_group(f, "Set1")
    dset = create_dataset(gr, "Parameters", Float64, (NParams, NSamples))
    write(dset, paramsList_check)
    dset = create_dataset(gr, "G0W", ComplexF64, (Nν, NSamples))
    write(dset, G0WList)
    dset = create_dataset(gr, "GImp", ComplexF64, (Nν, NSamples))
    write(dset, GImpList)
    dset = create_dataset(gr, "SImp", ComplexF64, (Nν, NSamples))
    write(dset, ΣImpList)
    dset = create_dataset(gr, "GL", ComplexF64, (NCoeffs, NSamples))
    write(dset, GLList)
    dset = create_dataset(gr, "SL", ComplexF64, (NCoeffs+1, NSamples))
    write(dset, SLList)
    dset = create_dataset(gr, "dens", Float64, (NSamples,))
    write(dset, densList)

end
