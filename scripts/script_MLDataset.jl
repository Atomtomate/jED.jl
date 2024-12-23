using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(joinpath(@__DIR__,".."))
@everywhere using jED
using HDF5

VkSamples = 2
EkSamples = 2
MuSamples = 2
betaSamples = 1
USamples = 1

NBath = 4

βList = [30.0] # 1 ./ LinRange(0.06,1,betaSamples)
UList = [1.0]

println("Total length: ", USamples*betaSamples*MuSamples*VkSamples^NBath*EkSamples^NBath)

Ui = UList[1]
V1 = LinRange(0,1.0,VkSamples)
E1 = LinRange(-2Ui, 2Ui, EkSamples)
μList = LinRange(-Ui, 2*Ui, MuSamples) 

fullParamList = collect(Base.product(E1,E1,E1,E1,V1,V1,V1,V1,μList,UList,βList))[:]
NSamples = length(fullParamList)
println("check: ", NSamples)



@everywhere function solve_imp(parList; Nν::Int = 100, NB::Int = 4, β::Float64 = 30.0, dens_eps::Float64 = 1e-1)
    # tsc= 0.40824829046386307/2      # hopping amplitude
    # Nk::Int = 40                         # Number of k-sampling points i neach direction
    # kG = jED.gen_kGrid("3Dsc-$tsc", Nk)                 # struct for k-grid

    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    # ========== TODO: this is only fixed for this test ============

    basis  = jED.Basis(NB+1);       # ED basis
    imp_OP = create_op(basis, 1)    # Operator for impurity measurement
    overlap= Overlap(basis, imp_OP)       # helper for <i|OP|j> overlaps for 
    # Σ0 = similar(νnGrid); fill!(Σ0, 0.0)
    #GBath = GLoc(Σ0, μ, νnGrid, kG)

    maxS = length(parList)
    val_ind = falses(maxS)
    parList_p   = Matrix{Float64}(undef, length(parList[1]), maxS)
    G0WMatrix   = Matrix{ComplexF64}(undef, Nν, maxS)
    GImpMatrix = Matrix{ComplexF64}(undef, Nν, maxS)
    ΣImpMatrix = Matrix{ComplexF64}(undef, Nν, maxS)
    densList   = Vector{Float64}(undef, maxS)

    small = 0
    large = 0
    for (it, el) in enumerate(parList)
        ϵₖ = collect(el[1:NB])
        Vₖ = collect(el[NB+1:2*NB])
        μ = el[2*NB+1]
        U = el[2*NB+2]
        β = el[2*NB+3]
        p    = AIMParams(ϵₖ, Vₖ)          # Anderson Parameters (struct type)
        G0W  = GWeiss(νnGrid, μ, p)

        # Impurity solution
        model  = AIM(ϵₖ, Vₖ, μ, U)
        es     = Eigenspace(model, basis, verbose=false);
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-12, overlap=overlap, with_density=true)
        skip = false
        dens < dens_eps && (small += 1; skip = true)
        dens > 2 - dens_eps && (large += 1; skip = true)
        #ΣImp_i = Σ_from_GImp(GBath, GImp_i)
        #fit_AIM_params!(p, GImp_i, μ, νnGrid)
       
        if !skip
            val_ind[it] = true
            ΣImp_i = Σ_from_GImp(G0W, GImp_i)
            parList_p[:,it] = collect(el)
            G0WMatrix[:,it] = G0W

            GImpMatrix[:,it] = GImp_i
            ΣImpMatrix[:,it] = ΣImp_i
            densList[it] = dens
        end
    end
    println("small = $small ($(small/length(parList)))")
    println("large = $large ($(large/length(parList)))")
    return parList_p[:,val_ind], G0WMatrix[:,val_ind], GImpMatrix[:,val_ind], ΣImpMatrix[:,val_ind], densList[val_ind]
end

NWorkers = length(workers())
ChunkSize = ceil(Int, NSamples/NWorkers)
batch_indices = collect(Base.Iterators.partition(1:length(fullParamList),ChunkSize))
futures = []
for (i,wi) in enumerate(workers())
    push!(futures, remotecall(solve_imp, wi, fullParamList[batch_indices[i]]))
end

function run()
    Nν::Int = 100 
    NParams = length(fullParamList[1])
    paramsList_check = Array{Float64,2}(undef, NParams, 0)
    G0WList = Array{ComplexF64,2}(undef, Nν, 0)
    GImpList = Array{ComplexF64,2}(undef, Nν, 0)
    ΣImpList = Array{ComplexF64,2}(undef, Nν, 0)
    densList = Array{Float64,1}(undef, 0)

    for (i,res) in enumerate(workers())
        println("optaining results from worker $i")
        res = fetch(futures[i])
        ind = batch_indices[i]
        #println(res[1])
        #println(size(res[1]))
        #println(size(res[2]))
        paramsList_check = hcat(paramsList_check, res[1]) 
        G0WList = hcat(G0WList, res[2])
        GImpList = hcat(GImpList, res[3]) 
        ΣImpList= hcat(ΣImpList, res[4])
        push!(densList, res[5]...)
    end
    NSamples_pruned = length(densList)
    println("disregarded $(100 - 100*NSamples_pruned/NSamples) % of generated samples")

    fn = "data_batch1_NB4_nPrune.hdf5"
    h5open(fn, "w") do f
        gr = create_group(f, "Set1")
        dset = create_dataset(gr, "Parameters", Float64, (NParams, NSamples_pruned))
        write(dset, paramsList_check)
        dset = create_dataset(gr, "G0W", ComplexF64, (Nν, NSamples_pruned))
        write(dset, G0WList)
        dset = create_dataset(gr, "GImp", ComplexF64, (Nν, NSamples_pruned))
        write(dset, GImpList)
        dset = create_dataset(gr, "SImp", ComplexF64, (Nν, NSamples_pruned))
        write(dset, ΣImpList)
        dset = create_dataset(gr, "dens", Float64, (NSamples_pruned,))
        write(dset, densList)
        dset = create_dataset(gr, "runscript", String, (1,))
        write(dset, read(@__FILE__, String))
    end
end

#run()
print(String(@__FILE__))
rmprocs()
