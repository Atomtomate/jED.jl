using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using TimerOutputs
using JLD2
to = TimerOutput()

β  = 19.4
t = 1.0


GFs = []
energies = []
Z_list = []
E_list = []
evals_list = []
params = []
UList = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
NSitesList = 2:2:10
for U in UList
    for NSites in NSitesList
        global GFs
        global energies
        global Z_list
        global E_list
        println("U=$U; NSites= $NSites")
        basis = jED.Basis(NSites, N_filter=[NSites], S_filter=[0])
        model = Hubbard_Full(-t, U, NSites)

        es = Eigenspace(model, basis);

        #νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:80], 0:80)
        #GF, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-12, with_density=true)
        Z = calc_Z(es, β)
        E = calc_E(es, β)
        #push!(GFs, GF)
        push!(energies, es.E0)
        push!(Z_list, Z)
        push!(E_list, E)
        push!(evals_list, es.evals)
        push!(params, (U,NSites))
        println("E₀ = $(es.E0) ")
    end
end
#println("Diagonalized Hamiltonian")
#show_diag_Hamiltonian(basis, es)


#println("Energies (sorted in blocks)")
#show_energies_states(basis, es)
pls = []
for i in 1:length(UList)
    ind = 1 + ((i-1)*length(NSitesList))
    norm = params[ind][2]#params[ind][2]*params[ind][1] #Z_list[ind]# maximum(evals_list[ind]) #Z_list[ind]
    bins = range(0, ceil(Int,maximum(evals_list[ind] ./ norm)), length=10)
    # 
    pli = stephist(evals_list[ind] ./ norm, normalize=:pdf, alpha=1.2, label="U=$(params[ind][1]), NSites=$(params[ind][2])", size=(1400,800), legend=:outerleft, bins=bins)
    for Ni in 2:1:length(NSitesList)
        
        ind = Ni + ((i-1)*length(NSitesList))
        norm = params[ind][2]#Z_list[ind]# maximum(evals_list[ind])
        println(ind)
        stephist!(pli, evals_list[ind] ./ norm, alpha=1.2, normalize=true, label="U=$(params[ind][1]), NSites=$(params[ind][2])", bins=bins)
    end
    push!(pls, pli)
  end

plot(pls...)


pls2 = []
for Ni in 1:1:length(UList)
    ind = 1 + ((Ni-1)*length(NSitesList))
    norm = params[ind][2]#params[ind][2]*params[ind][1] #Z_list[ind]# maximum(evals_list[ind]) #Z_list[ind]
    bins = range(0, ceil(Int,maximum(evals_list[ind] ./ norm)), length=10)
    # 
    pli = histogram(evals_list[ind] ./ norm, normalize=:pdf, alpha=1.0 - 0.2*1, label="U=$(params[ind][1]), NSites=$(params[ind][2])", size=(1400,800), legend=:outerleft, bins=bins)
    for i in 2:1:length(NSitesList)
        ind = i + ((Ni-1)*length(NSitesList))
        norm = params[ind][2]#Z_list[ind]# maximum(evals_list[ind])
        println(ind)
        histogram!(pli, evals_list[ind] ./ norm, alpha=1.0 - 0.2*i, normalize=true, label="U=$(params[ind][1]), NSites=$(params[ind][2])", bins=bins)
    end
    push!(pls2, pli)
  end

plot(pls2...)