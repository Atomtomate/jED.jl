using Pkg
Pkg.activate(@__DIR__)
using jED
using NLsolve
using FiniteDiff
using NLSolvers
using LsqFit
using Memoization

ϵₖ = [1.0, 0.5, -1.1, -0.6]
Vₖ = [0.25, 0.35, 0.45, 0.55]
p  = AIMParams(ϵₖ, Vₖ)
μ  = 0.6
U  = 1.2
β  = 4.0
tsc= 0.40824829046386307/2
Nν = 1000
Nk = 20
α  = 0.2
GImp_i = nothing
GImp_i_old = nothing

kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)
basis  = jED.Basis(length(Vₖ) + 1);
νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
N =   length(ϵₖ)

function DMFT_solve_test(p_vec::Vector)::Float64
    p      = AIMParams(p_vec)
    model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
    G0W    = GWeiss(νnGrid, μ, p)
    es     = Eigenspace(model, basis);
    println("     Calculating GImp")
    GImp_i = calc_GF_1(basis, es, νnGrid, β)
    ΣImp_i = Σ_from_GImp(G0W, GImp_i)
    GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
    fit_AIM_params!(p, GLoc_i, μ, νnGrid)
    res = sum(abs.(vcat(p.ϵₖ, p.Vₖ) .- p_vec))
    println("Error = $res, $(p.ϵₖ) , $(p.Vₖ)")
    return res
end

objective(x) = DMFT_solve_test(x)
function grad(∇f, x) 
    ∇f = FiniteDiff.finite_difference_gradient(objective,x)
    return ∇f
end
function hess(∇²f, x) 
    ∇²f = FiniteDiff.finite_difference_hessian(objective,x)
    return ∇²f
end
objective_grad(∇f,x) = objective(x), grad(∇f, x)
function objective_grad_hess(∇f,∇²f,x) 
    f, ∇f = objective_grad(∇f,x) 
    ∇²f   = hess(∇²f, x)
    return f,∇f,∇²f
end 

# Solve using NLSolvers (see: github.com/JuliaNLSolvers/NLSolvers.jl)
scalarobj = ScalarObjective(f=objective, g=grad, h=hess,fg=objective_grad, fgh=objective_grad_hess)                 # define objective function, here: GLOC === G0W
optprob   = OptimizationProblem(scalarobj; inplace=false)  # define optimiztion problem, here scalar function

#  ==== Solve, using CG
println("Solving with CG")
p0        = vcat(p.ϵₖ, p.Vₖ)
res_CG       = solve(optprob, p0, ConjugateGradient(), OptimizationOptions(f_abstol=1e-4, maxiter=50))                                           # solve problem
p_CG = res_CG.info.solution

#  ==== Solve, using Newton
println("Solving with Newton")
res_Newton       = solve(optprob, p0, LineSearch(Newton()), OptimizationOptions(f_abstol=1e-4, maxiter=50))                                           # solve problem
p_Newton = res_Newton.info.solution
println("Solution using CG:     ϵₖ = $(lpad.(round.(p_CG[1:N],digits=4),9)...)")
println("                       Vₖ = $(lpad.(round.(p_CG[N+1:end],digits=4),9)...)")
println(" -> sum(Vₖ²) = $(sum(p_CG[N+1:end] .^ 2))")
println("Solution using Newton: ϵₖ = $(lpad.(round.(p_Newton[1:N],digits=4),9)...)")
println("                       Vₖ = $(lpad.(round.(p_Newton[N+1:end],digits=4),9)...)")
println(" -> sum(Vₖ²) = $(sum(p_Newton[N+1:end] .^ 2))")
