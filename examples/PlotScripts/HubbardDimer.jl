using Plots
using LaTeXStrings
using Pkg
Pkg.activate(joinpath(@__DIR__,"../.."))
using jED
using TimerOutputs

to = TimerOutput()

β  = 19.4

t = 2.0
U = 100.0
energyList= []
ZList= []
EList= []
UList = LinRange(0,20.0,50)
for U in UList
    global energyList
    global ZList
    global EList
    tMatrix = [0.0 t
            t 0.0]
    UMatrix = [U   0.0
            0.0 U]
    basis = jED.Basis(2, N_filter=[2], S_filter=[0])
    model = Hubbard(tMatrix, UMatrix)
    es = Eigenspace(model, basis);
    Z = calc_Z(es, β)
    E = calc_E(es, β)
    E₀ = U/2 - sqrt((U^2) + 16 * t^2)/2
    E₁ = U/2 + sqrt((U^2) + 16 * t^2)/2
    tt = es.evals .+ es.E0
    push!(energyList, tt)
    push!(ZList, Z)
    push!(EList, E)
end

p = plot(UList, map(x->x[1],energyList),  lw=3, label=L"U_\mathrm{low}",xlabel=L"U", legend=:outerright,
         yguidefontsize=16,legendfontsize=16,xtickfontsize=16,ytickfontsize=16)
#plot!(UList, map(x->x[2],energyList), lw=2, label=L"0")
#plot!(UList, map(x->x[3],energyList), lw=2,  label=L"U")
plot!(UList, map(x->x[4],energyList), lw=3, label=L"U_\mathrm{high}")

plot!(UList, UList .+  ((4 .* (t^2)) ./ UList), lw=3, ls=:dash, label=L"U + J")#, ylims=(-1,0.4)
plot!(UList, - (4 .* t^2) ./ UList, lw=3,ls=:dash, label=L"\;\;\;-J")
savefig("Hubbard_To_Heisenberg.pdf")