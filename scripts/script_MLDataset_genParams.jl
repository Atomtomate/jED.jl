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