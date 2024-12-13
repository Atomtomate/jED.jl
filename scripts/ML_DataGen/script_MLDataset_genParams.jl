using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(joinpath(@__DIR__,".."))
@everywhere using jED
@everywhere using JLD2

VkSamples = 12
EkSamples = 12
MuSamples = 12
betaSamples = 1
USamples = 1

NBath = 3
add_noise = true

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