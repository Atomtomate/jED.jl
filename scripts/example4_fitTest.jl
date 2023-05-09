using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using NLsolve
using FiniteDiff
using NLSolvers
using LsqFit


ϵₖ = [1.0, 0.5, -1.1, -0.6]
Vₖ = [0.25, 0.35, 0.45, 0.55]
p  = AIMParams(ϵₖ, Vₖ)
μ  = 0.6
U  = 1.2
β  = 4.0
tsc= 0.40824829046386307/2
Nν = 1000
Nk = 40

kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)
basis  = jED.Basis(5);
νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
G0W    = GWeiss(νnGrid, μ, p)

model  = AIM(ϵₖ, Vₖ, μ, U)
es     = Eigenspace(model, basis);
GImp_i, dens = calc_GF_1(basis, es, νnGrid, β)
ΣImp_i = Σ_from_GImp(G0W, GImp_i)

GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
G0W    = GWeiss_from_Imp(GLoc_i, ΣImp_i)
#fit_AIM_params!(p, GLoc_i, μ, νnGrid)

tmp = similar(GLoc_i.parent) 
N =   length(ϵₖ)
function objective(p::Vector)
    GWeiss!(tmp, νnGrid.parent, μ, p[1:N], p[N+1:end])
    sum(abs.(tmp .- GLoc_i.parent))/length(tmp)
end
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

p0        = vcat(p.ϵₖ, p.Vₖ)
res       = solve(optprob, p0, ConjugateGradient(), OptimizationOptions(f_abstol=1e-8, maxiter=50))                                           # solve problem
p_CG = res.info.solution

res       = solve(optprob, p0, LineSearch(Newton()), OptimizationOptions(f_abstol=1e-8, maxiter=50))                                           # solve problem
p_Newton = res.info.solution

# Solve using LsqFit
function GW_fit_real(νnGrid::Vector, p::Vector)::Vector{Float64} 
    GWeiss!(tmp, νnGrid, μ, p[1:N], p[(N+1):end])
    vcat(real(tmp), imag(tmp))
end
target   = vcat(real(GLoc_i.parent),imag(GLoc_i.parent))
res_Lsq  = curve_fit(GW_fit_real, νnGrid.parent, target, p0)
p_Lsq    = res_Lsq.param

println("Solution using CG:     ϵₖ = $(lpad.(round.(p_CG[1:N],digits=4),9)...)")
println("                       Vₖ = $(lpad.(round.(p_CG[N+1:end],digits=4),9)...)")
println(" -> sum(Vₖ²) = $(sum(p_CG[N+1:end]))")
println("Solution using Newton: ϵₖ = $(lpad.(round.(p_Newton[1:N],digits=4),9)...)")
println("                       Vₖ = $(lpad.(round.(p_Newton[N+1:end],digits=4),9)...)")
println(" -> sum(Vₖ²) = $(sum(p_Newton[N+1:end]))")
println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p_Lsq[1:N],digits=4),9)...)")
println("                       Vₖ = $(lpad.(round.(p_Lsq[N+1:end],digits=4),9)...)")
println(" -> sum(Vₖ²) = $(sum(p_Lsq[N+1:end]))")
