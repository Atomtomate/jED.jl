using Pkg
Pkg.activate(@__DIR__)
using jED
using NLsolve
using FiniteDiff
using NLSolvers
using LsqFit


function DMFT_Loop_test(;maxit = 20)
    ϵₖ = [1.0, 0.5, -1.1, -0.6]
    Vₖ = [0.25, 0.35, 0.45, 0.55]
    p  = AIMParams(ϵₖ, Vₖ)
    μ  = 0.6
    U  = 1.2
    β  = 4.0
    tsc= 0.25
    Nν = 1000
    Nk = 40
    α  = 0.4
    GImp_i = nothing
    GImp_i_old = nothing

    kG     = jED.gen_kGrid("Hofstadter:2:3-$tsc", Nk)
    basis  = jED.Basis(length(Vₖ) + 1);
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)

    for i in 1:maxit
        println("    ========== ITERATION $(rpad(i,3)) ==========    ")

        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        GImp_i = calc_GF_1(basis, es, νnGrid, β)
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        println("     Calculating GLoc")
        GLoc_i = jED.GLoc_MO(ΣImp_i, μ, νnGrid, kG)

        #  ==== Define objective function, gradient and hessian
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

        #  ==== Solve, using CG
        p0        = vcat(p.ϵₖ, p.Vₖ)
        res       = solve(optprob, p0, ConjugateGradient(), OptimizationOptions(f_abstol=1e-8, maxiter=50))                                           # solve problem
        p_CG = res.info.solution

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
        println(" -> sum(Vₖ²) = $(sum(p_CG[N+1:end] .^ 2))")
        println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p_Lsq[1:N],digits=4),9)...)")
        println("                       Vₖ = $(lpad.(round.(p_Lsq[N+1:end],digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p_Lsq[N+1:end] .^ 2))")

        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println("     iteration $i with AIM params ∑ Vₖ^2 = $(sum(p.Vₖ .^ 2)), checksum GImp = $(abs(sum(GImp_i)))")
        
    end
end

DMFT_Loop_test(maxit = 20)
