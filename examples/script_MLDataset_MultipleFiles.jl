using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(joinpath(@__DIR__,".."))
@everywhere using jED
@everywhere using JLD2

VkSamples = 35
EkSamples = 35
MuSamples = 20
betaSamples = 1
USamples = 1

NBath = 3

βList = [30.0] # 1 ./ LinRange(0.06,1,betaSamples)
UList = [1.0]

println("Total length: ", USamples*betaSamples*MuSamples*VkSamples^NBath*EkSamples^NBath)

Ui = UList[1]
V1 = LinRange(0,1.0,VkSamples)
E1 = LinRange(-2Ui, 2Ui, EkSamples)
μList = LinRange(-Ui, 2*Ui, MuSamples) 

fullParamList = collect(Base.product(repeat([E1],NBath)...,repeat([V1],NBath)...,μList,UList,βList))[:]
NSamples = length(fullParamList)
println("check: ", NSamples)

@everywhere function solve_imp(parList, fn_name::String, index::Int; Nν::Int = 100, NB::Int = 2, β::Float64 = 30.0, dens_eps::Float64 = 1e-1)
    # tsc= 0.40824829046386307/2      # hopping amplitude
    # Nk::Int = 40                         # Number of k-sampling points i neach direction
    # kG = jED.gen_kGrid("3Dsc-$tsc", Nk)                 # struct for k-grid

    # ========== TODO: this is only fixed for this test ============
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    # ========== TODO: this is only fixed for this test ============

    basis  = jED.Basis(NB+1);       # ED basis
    imp_OP = create_op(basis, 1)    # Operator for impurity measurement
    overlap= Overlap(basis, imp_OP)       # helper for <i|OP|j> overlaps for 
    # Σ0 = similar(νnGrid); fill!(Σ0, 0.0)
    #GBath = GLoc(Σ0, μ, νnGrid, kG)


    G0WMatrix   = Matrix{ComplexF64}(undef, Nν, length(parList))
    GImpMatrix = Matrix{ComplexF64}(undef, Nν, length(parList))
    ΣImpMatrix = Matrix{ComplexF64}(undef, Nν, length(parList))
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
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-12, overlap=overlap, with_density=true)

        skip = false
        dens < dens_eps && (skip = true)
        dens > 2 - dens_eps && (skip = true)
        #ΣImp_i = Σ_from_GImp(GBath, GImp_i)
        #fit_AIM_params!(p, GImp_i, μ, νnGrid)
       
        if !skip
            ΣImp_i = Σ_from_GImp(G0W, GImp_i)
            G0WMatrix[:,it] = G0W
            GImpMatrix[:,it] = GImp_i
            ΣImpMatrix[:,it] = ΣImp_i
            densList[it] = dens
        end
    end
    NSamples = length(parList)
    fn = fn_name*"_$index.hdf5"
    jldopen(fn, "w") do f
        f["Set1/Parameters"] = hcat(collect.(parList)...)
        f["Set1/G0W"] = G0WMatrix
        f["Set1/GImp"] = GImpMatrix
        f["Set1/dens"] = densList
        f["Set1/SImp"] = ΣImpMatrix
        f["Set1/runscript"] = read(@__FILE__, String)
    end
    return nothing
end

NChunks = 30
#NWorkers = 30#length(NChunks)
NSamples = length(fullParamList)
ChunkSize = ceil(Int, NSamples/NChunks)
batch_indices = collect(Base.Iterators.partition(1:length(fullParamList),ChunkSize))
futures = []
wp = WorkerPool(workers())

for (i,wi) in enumerate(1:NChunks)
    push!(futures, remotecall(solve_imp, wp, fullParamList[batch_indices[i]], "batch3_nPrune", i))
end

for (i,res) in enumerate(1:NChunks)
    wait(futures[i])
end


