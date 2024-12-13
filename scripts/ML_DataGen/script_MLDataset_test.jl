using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using Statistics 

VkSamples = 10
EkSamples = 10
MuSamples = 20
betaSamples = 1
USamples = 1

NBath = 2
add_noise = true

βList = [30.0] # 1 ./ LinRange(0.06,1,HubbParSamples)
UList = [1.0]

println("Total length: ", USamples*betaSamples*MuSamples*VkSamples^NBath*EkSamples^NBath)

Ui = UList[1]
V1 = LinRange(0,1.0,VkSamples)
E1 = LinRange(-2Ui, 2Ui, EkSamples)
μList = LinRange(-2*Ui, 2*Ui, MuSamples) 

fullParamList = collect(Base.product(E1,E1,V1,V1,μList,UList,βList))[:]
if add_noise
    V_noise_level = mean(diff(V1)) / 10
    E_noise_level = mean(diff(E1)) / 10
    μ_noise_level = mean(diff(μList)) / 10
    for i in eachindex(fullParamList)
        noise_E = randn(NBath) .* E_noise_level
        noise_V = randn(NBath) .* V_noise_level
        noise_μ = randn()  * μ_noise_level
        fullParamList[i] = fullParamList[i] .+ (noise_E..., noise_V..., noise_μ, 0.0, 0.0)
    end
end



Nν::Int = 100
NB::Int = 2
β::Float64 = 30.0
νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
basis  = jED.Basis(NB+1);       # ED basis
imp_OP = create_op(basis, 1)    # Operator for impurity measurement
overlap= Overlap(basis, imp_OP)       # helper for <i|OP|j> overlaps for 

e1, e2, v1, v2, μ, U, β = fullParamList[2013]
ϵₖ = [e1, e2]
Vₖ = [v1, v2]

p    = AIMParams(ϵₖ, Vₖ)          # Anderson Parameters (struct type)
G0W  = GWeiss(νnGrid, μ, p)

# Impurity solution
model  = AIM(ϵₖ, Vₖ, μ, U)
es     = Eigenspace(model, basis, verbose=false);
GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-12, overlap=overlap, with_density=true)
ΣImp_i = Σ_from_GImp(G0W, GImp_i)
