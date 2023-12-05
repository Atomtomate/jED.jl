using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED

tsc= 0.40824829046386307/2      # hopping amplitude
Nk = 40                         # Number of k-sampling points i neach direction
kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)                 # struct for k-grid

VkSamples = 10
HubbParSamples = 10
NBath = 2


μ  = 0.6
U  = 1.2
β  = 4.0
Nν = 1000

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
