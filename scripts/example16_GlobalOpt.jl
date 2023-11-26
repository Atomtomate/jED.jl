using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using TimerOutputs
using NLopt
to = TimerOutput()

FPT = jED.Float64

algorithms = [:DIRECT, :LD_MMA]


# for alg in algorithms
#     pass
# end

function DMFT_Loop(U::Float64, μ::Float64, β::Float64; maxit = 20)
    ϵₖ = [1.0, 0.5, -1.0, -0.5]#, 1.0, -1.0]
    Vₖ = [0.25, 0.35, 0.25, 0.35]#, 0.6, 0.7]

    NBath::Int = floor(Int, length(ϵₖ))
    p  = AIMParams(ϵₖ, Vₖ)
    tsc = 0.25
    Nν  = 1000
    Nk  = 40
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

        function objective_f(x::Vector, grad::Vector)
            tmp = similar(νnGrid.parent)
            GWeiss!(tmp, νnGrid, μ, x[1:NBath], x[(NBath+1):end])
            return sum(abs.(tmp .- GLoc_i))
        end

        function myconstraint(x::Vector, grad::Vector, a, b)
            sum(abs.(x[1:NBath]) .- 5*U) + abs(sum(x[(NBath+1):end] .^2 )) - 0.3
        end

        alg = :LD_MMA

        opt = Opt(alg, 8)
        opt.min_objective = objective_f
        inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
        inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

        (minf,minx,ret) = optimize(opt, [p.ϵₖ..., p.Vₖ...])
        numevals = opt.numevals # the number of function evaluations
        println("got $minf at $minx after $numevals iterations (returned $ret)")

        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2)), density = $dens")

        println("Solution using $alg: ")
        println("         ϵₖ = $(lpad.(round.(minx[1:NBath],digits=4),9)...)")
        println("         Vₖ = $(lpad.(round.(minx[NBath+1:end],digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2)), density = $dens")
    end
    return p, νnGrid, GImp_i, ΣImp_i
end

#                 ]
p, νnGrid, GImp, ΣImp = DMFT_Loop(2.0, 1.0, 1.0; maxit = 10)

