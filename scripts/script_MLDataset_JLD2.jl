using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(joinpath(@__DIR__,".."))
@everywhere using jED
using JLD2

VkSamples = 2
EkSamples = 2
MuSamples = 2
betaSamples = 1
USamples = 1

NBath = 2

βList = [30.0] # 1 ./ LinRange(0.06,1,HubbParSamples)
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
    # tsc= 0.40824829046386307/2      # hopping amplitude
    # Nk::Int = 40                         # Number of k-sampling points i neach direction
    # kG = jED.gen_kGrid("3Dsc-$tsc", Nk)                 # struct for k-grid

    # ========== TODO: this is only fixed for this test ============
    Nν::Int = 100
    NB::Int = 2
    β::Float64 = 30.0
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
        #ΣImp_i = Σ_from_GImp(GBath, GImp_i)
        #fit_AIM_params!(p, GImp_i, μ, νnGrid)
       
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        # GLoc_i = jED.GLoc(ΣImp_i, μ, νnGrid, kG)
        # # Fits
        # fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        G0WMatrix[:,it] = G0W
        GImpMatrix[:,it] = GImp_i
        ΣImpMatrix[:,it] = ΣImp_i
        densList[it] = dens
    end
    return hcat(collect.(parList)...), G0WMatrix, GImpMatrix, ΣImpMatrix, densList
end

NWorkers = length(workers())
ChunkSize = ceil(Int, NSamples/NWorkers)
batch_indices = collect(Base.Iterators.partition(1:length(fullParamList),ChunkSize))
futures = []
for (i,wi) in enumerate(workers())
    push!(futures, remotecall(solve_imp, wi, fullParamList[batch_indices[i]]))
end


Nν::Int = 100 
NParams = length(fullParamList[1])
paramsList_check = Array{Float64,2}(undef, NParams, NSamples)
G0WList = Array{ComplexF64,2}(undef, Nν, NSamples)
GImpList = Array{ComplexF64,2}(undef, Nν, NSamples)
ΣImpList = Array{ComplexF64,2}(undef, Nν, NSamples)
densList = Array{Float64,1}(undef, NSamples)
fill!(densList, NaN)

for (i,res) in enumerate(workers())
    res = fetch(futures[i])
    ind = batch_indices[i]
    println(res[1])
    println(size(res[1]))
    println(size(res[2]))
    paramsList_check[:, ind] = res[1] 
    G0WList[:, ind] = res[2] 
    GImpList[:, ind] = res[3] 
    ΣImpList[:, ind] = res[4] 
    densList[ind] = res[5] 
end

fn = "test.jld2"
jldopen(fn, "w") do f
    f["Set1/Parameters"] = paramsList_check
    f["Set1/G0W"] = G0WList
    f["Set1/GImp"] = GImpList
    f["Set1/dens"] = densList

end
