using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using TimerOutputs
using NLopt
to = TimerOutput()

FPT = jED.Float64

# crashes: 
# invalid args:  :LD_LBFGS_NOCEDAL, :LD_AUGLAG, :LD_AUGLAG_EQ, 
# slow: 
# Local Algorithms
algorithms_LD = [:LD_CCSAQ, :LD_LBFGS, 
                 :LD_MMA, :LD_SLSQP, :LD_TNEWTON, :LD_TNEWTON_PRECOND,
                :LD_TNEWTON_PRECOND_RESTART, :LD_TNEWTON_RESTART, :LD_VAR1, :LD_VAR2]

# Global Algorithms
algorithms_GN = [:GN_CRS2_LM, :GN_DIRECT, :GN_DIRECT_L,
                 :GN_DIRECT_L_NOSCAL, :GN_DIRECT_L_RAND, :GN_DIRECT_L_RAND_NOSCAL,
                 :GN_DIRECT_NOSCAL, :GN_ESCH, :GN_MLSL,
                 :GN_MLSL_LDS,]
algorithms_GN_constraints = [:GN_AGS, :GN_ISRES, :GN_ORIG_DIRECT, :GN_ORIG_DIRECT_L]

#algorithms = vcat(algorithms_GN, algorithms_LD)
#algorithms = algorithms_GN_constraints
algorithms = vcat(algorithms_GN, algorithms_GN_constraints)

results = Dict{Int,Dict{Symbol, Tuple{Symbol, Vector}}}()


MTime = 1.0
println("Comparing algorithms, giving $MTime [s] time allocation for each of them. Change MTime, if you want to investigate full convergence behavior.")

function DMFT_Loop(U::Float64, μ::Float64, β::Float64; maxit = 20)
    # deliberately starting with bad initial guess
    ϵₖ = [0.1, 0.2, 0.3, 0.4]#, 1.0, -1.0]
    Vₖ = [0.1, 0.2, 0.3, 0.4]#, 0.6, 0.7]

    NBath::Int = floor(Int, length(ϵₖ))
    p  = AIMParams(ϵₖ, Vₖ)
    tsc = 0.25
    Nν  = 200
    Nk  = 20
    α   = 0.4
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing

    kG     = jED.gen_kGrid("2Dsc-$tsc", Nk)
    basis  = jED.Basis(NBath + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.FermionicMatsubaraGrid{FPT}([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
        
    for i in 1:maxit
        println("    ========== ITERATION $(rpad(i,3)) ==========    ")
        model  = AIM{FPT}(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis);
        #isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        @timeit to "GImp" GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, overlap=overlap, with_density=true)
        #!isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)

        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2)), density = $dens")


        function objective_f(x::Vector, grad::Vector)
            tmp = similar(νnGrid.parent)
            jED.GWeiss!(tmp, νnGrid.parent, μ, x[1:NBath], x[(NBath+1):end])
            return sum(abs.(tmp .- GLoc_i.parent))
        end

        function myconstraint(x::Vector, grad::Vector, a, b)
            sum(abs.(x[1:NBath]) .- 5*U) + abs(sum(x[(NBath+1):end] .^2 )) - 0.3 + 1 / sum(abs.(x[1:NBath]))
        end

        xvec_init = [p.ϵₖ..., p.Vₖ...]
        println("Check for initial values and constraints:")
        println("    -> ", objective_f(xvec_init, []))
        println("    -> ", myconstraint(xvec_init, [], NaN, NaN))
        results[i] = Dict{Symbol, Tuple{Symbol, Vector}}()

        for alg in algorithms
            opt = Opt(alg, 2*NBath)
            opt.lower_bounds = [repeat([-3U],NBath)..., repeat([-sqrt(0.25)], NBath)...]
            opt.upper_bounds = [repeat([3U],NBath)..., repeat([sqrt(0.25)], NBath)...]
            opt.min_objective = objective_f
            opt.ftol_abs = 1e-8 
            opt.xtol_abs = 1e-8 
            opt.maxtime  = MTime
            if alg in algorithms_GN_constraints
                inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
                inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)
            end

            @timeit to "opt $alg" (minf,minx,ret) = optimize(opt, xvec_init)
            numevals = opt.numevals # the number of function evaluations
            println("   -------------------------------------------------------- ")
            println("                           $alg")
            println("   -------------------------------------------------------- ")
            println("got $minf at $(round.(minx,digits=3)) after $numevals iterations (returned $ret)")

            println("         ϵₖ = $(lpad.(round.(minx[1:NBath],digits=4),9)...)")
            println("         Vₖ = $(lpad.(round.(minx[NBath+1:end],digits=4),9)...)")
            println(" -> sum(Vₖ²) = $(sum(minx[NBath+1:end] .^ 2))")
            println("   -------------------------------------------------------- ")
            results[i][alg] = (ret, minx)

        end
    end
    return results
end

#                 ]
results = DMFT_Loop(2.0, 1.0, 1.0; maxit = 5)
println(to)

# printing results

for i in sort(collect(keys(results)))
    println("    =============== ITERATION $(rpad(i,3)) ===============    ")
    for alg in keys(results[i])
        println("    ------------- $alg  ------------- ")
        println("         return status: ", results[i][alg][1])
        NBath::Int = floor(Int, length(results[i][alg][2])/2)
        ek = round.(results[i][alg][2][1:NBath], digits=3)
        Vk = round.(results[i][alg][2][NBath+1:end], digits=3)
        println("         return value : ek = ", ek, ", Vk = ", Vk)
    end
end
