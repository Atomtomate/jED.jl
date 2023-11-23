using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using Distributed

VkSamples = 10
HubbParSamples = 10
NBath = 2



fullParamList = []
sizehint!(fullParamList, HubbParSamples^2 * VkSamples^4)

βList  = 1 ./ LinRange(0.06,1,HubbParSamples)
UList  = LinRange(0.1, 4, HubbParSamples)
VKList = Vector{Vector}(undef, 0)
V1 = LinRange(1/VkSamples,0.5,VkSamples)
for V1i in V1
    V1i ≈ 0 && continue
    for V2i in LinRange(V1i,0.5, VkSamples+1)[2:end]
        V2i ≈ -0 && continue
        t = [V1i, V2i]
        push!(VKList, t ./ (2 * sqrt(sum(t .^ 2))))
    end
end

for βi in βList
    for Ui in UList
        for μi in [Ui/2]
            for E1i in LinRange(1/VkSamples,2*Ui,VkSamples)
                E1i ≈ 0 && continue
                for E2i in LinRange(E1i,0.5, VkSamples+1)[2:end]
                    E2i ≈ 0 && continue
                    for Vki in VKList
                        push!(fullParamList, [βi, Ui, μi, E1i, E2i, Vki[1], Vki[2]])
                    end
                end
            end
        end
    end
end

@everywhere function solve_imp(parList)
    tsc= 0.40824829046386307/2      # hopping amplitude
    Nk::Int = 40                         # Number of k-sampling points i neach direction
    kG = jED.gen_kGrid("3Dsc-$tsc", Nk)                 # struct for k-grid
    Nν::Int = 100
    NB::Int = 2
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)

    basis  = jED.Basis(NB+1);       # ED basis
    imp_OP = create_op(basis, 1)    # Operator for impurity measurement
    overlap= Overlap(basis, imp_OP)       # helper for <i|OP|j> overlaps for 
    Σ0 = similar(νnGrid); fill!(Σ0, 0.0)
    GBath = GLoc(Σ0, μ, νnGrid, kG)


    GImpMatrix = Matrix{ComplexF64}(undef, Nν, length(parList))
    ΣImpMatrix = Matrix{ComplexF64}(undef, Nν, length(parList))
    pOutMatrix = Matrix{Float64}(undef, 4, length(parList))

    for (it, el) in enumerate(parList)
        β, U, μ, e1, e2, v1, v2 = el
        ϵₖ = [e1, e2]
        Vₖ = [v1, v2]

        p  = AIMParams(ϵₖ, Vₖ)          # Anderson Parameters (struct type)
        G0W    = GWeiss(νnGrid, μ, p)

        # Impurity solution
        model  = AIM(ϵₖ, Vₖ, μ, U)
        es     = Eigenspace(model, basis, verbose=false);
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-12, overlap=overlap, with_density=true)
        #ΣImp_i = Σ_from_GImp(GBath, GImp_i)
        #fit_AIM_params!(p, GImp_i, μ, νnGrid)
       
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        GLoc_i = jED.GLoc(ΣImp_i, μ, νnGrid, kG)

        # Fits
        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        GImpMatrix[:,it] = GImp_i
        ΣImpMatrix[:,it] = ΣImp_i
        pOutMatrix[:,it] = [p.ϵₖ...,p.Vₖ...]
    end
    return GImpMatrix, ΣImpMatrix, pOutMatrix
end
