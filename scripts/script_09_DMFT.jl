using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using ForwardDiff
using Optim, LsqFit
using TimerOutputs
using Logging
using JLD2

to = TimerOutput()
NSites = 4
β  = 40.0 #parse(Float64, ARGS[1])#75.0
kGStr = "2Dsc-0.25-0.075-0.05"# ARGS[2]
Nk = 200 #parse(Int, ARGS[3])
U  = 2.2 #parse(Float64, ARGS[4])
μ  = 0.6 #parse(Float64, ARGS[5])


function run_DMFT(NSites::Int, U::Float64, μ_in::Float64, β::Float64, fitf::Function; maxit = 20, verbose = true)
    Nν  = 2000
    α   = 0.4
    kG     = jED.gen_kGrid(kGStr,Nk) #"3Dsc-$tsc", Nk)
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    basis  = jED.Basis(NSites+1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional

    ϵₖ = vcat(randn(trunc(Int,NSites/2)) .* U .+ U,  randn(trunc(Int,NSites/2)) .* U .- U,  repeat([0.2], NSites%2))
    Vₖ = randn(NSites) ./ 100.0 .+ 0.17
    p  = AIMParams(ϵₖ, Vₖ)
    dens = NaN

    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing
    fit_res = nothing
    μ = μ_in
    fit_quality = nothing
    for i in 1:maxit
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis, verbose=false);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-15, overlap=overlap, with_density=false)
        Nup = calc_Nup(es, β, basis, model.impuritySiteIndex)
        Ndo = calc_Ndo(es, β, basis, model.impuritySiteIndex)
        dens = Nup + Ndo
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        !isnothing(GImp_i_old) && sum(abs.(GImp_i_old .- GImp_i)) < 1e-8 && break
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        fit_res = fitf(p, μ, GLoc_i, νnGrid)
        vals = Optim.converged(fit_res) ? Optim.minimizer(fit_res) : nothing
        isnothing(vals) && (@warn("Could not fit, aborting DMFT loop!"); break)
        p = AIMParams(vals[1:NSites], vals[NSites+1:end])
        sum_vk, min_eps_diff, min_vk, min_eps = andpar_check_values(vals[1:NSites], vals[NSites+1:end])
        fit_quality = [sum_vk, min_eps_diff, min_vk, min_eps ]

        tmp = 1 ./ (νnGrid.parent .+ μ .- sum((p.Vₖ .^ 2) ./ (reshape(νnGrid.parent,1,length(νnGrid)) .- p.ϵₖ), dims=1)[1,:])
        err = sum(abs.(GLoc_i.parent .- tmp))
        if verbose
            println(rpad("=========== Iteration $(i) =========",80,"="))
            println("   it[$i]: μ = $μ , dens = $dens")
            println("   Converged: ", Optim.converged(fit_res), " // Error = ", err, " // Error after trafo = ", Optim.minimum(fit_res))
            println("   1. min(|Vₖ|)      = ", min_vk)
            println("   2. ∑Vₗ^2          = ", sum_vk)
            println("   3. min(|ϵₖ|)      = ", min_eps)
            println("   4. min(|ϵₖ - ϵₗ|) = ", min_eps_diff)
            println("   Solution :    ϵₖ = $(lpad.(round.(vals[1:NSites],digits=4),9)...)")
            println("                 Vₖ = $(lpad.(round.(vals[NSites+1:end],digits=4),9)...)")
            println(repeat("=",80))
        end
    end
    return p, νnGrid, GImp_i, ΣImp_i, μ, dens, fit_res, fit_quality
end


transf_01(x, y) = y
transf_02(x, y) = 1 ./ y
transf_03(x, y) = y ./ x
transf_04(x, y) = x ./ y
transf_05(x, y) = 1 ./ (y .* x)
transf_06(x, y) = 1 ./ (y .* sqrt.(abs.(x)))
transforms_list  = [transf_01]#, transf_02, transf_03, transf_04, transf_05, transf_06]
transforms_names = ["y(x) → y(x)"]#, "y(x) → 1/y(x)", "y(x) → y(x)/x", "y(x) → x/y(x)", "y(x) → 1/(y(x)x)", "y(x) → 1/(y(x)√x)"]

optim_list  = [LBFGS()]
optim_names = ["LBFGS"]
opts = Optim.Options(iterations=20000,store_trace = false,
                             show_trace = false,
                             allow_f_increases = false,
                             x_tol = 1e-12,
                             f_tol = 1e-12,
                             show_warnings = true)

dist_list  = [jED.square_dist]#, jED.abs_dist]
dist_names = ["|vec|^2"]#, "|vec|"]

println("DBG: ", collect(zip(transforms_names, transforms_list)))

function run_tests(NSites, U, μ_in, β; maxit = 100, verbose=false)
    fits = []
    names = []
    i = 1
    global to
    for (transf_name, cf) in zip(transforms_names, transforms_list)
        for (opt_name,opt) in zip(optim_names, optim_list)
            for (dist_name, dist_f) in zip(dist_names, dist_list)
                run_name = "$cf  ($dist_name) // opt: $(typeof(opt))"
                println("running: ", run_name)
                function fitf(pAIM::AIMParams, μ, GLoc_i, νnGrid)
                    p0 = vcat(pAIM.ϵₖ, pAIM.Vₖ)
                    function wrap_cost(p::Vector)
                        GW_i = cf(νnGrid.parent, 1 ./ (νnGrid.parent .+ μ .- sum((p[NSites+1:end] .^ 2) ./ (reshape(νnGrid.parent,1,length(νnGrid)) .- p[1:NSites]), dims=1)[1,:]))
                        GW_i = vcat(real(GW_i),imag(GW_i))
                        GL_i = cf(νnGrid.parent, GLoc_i.parent)
                        GL_i = vcat(real(GL_i),imag(GL_i))
                        return dist_f(GL_i .- GW_i)
                    end
                    @timeit to "$run_name" result = optimize(wrap_cost, p0, opt, opts; autodiff = :forward)
                    return  result
                end
                p, νnGrid, GImp_i, ΣImp_i, μ, dens, fit_res, fit_quality  = run_DMFT(NSites, U, μ_in, β, fitf; maxit = maxit, verbose=verbose)
                println("  -> result: ", fit_res)
                push!(fits, (p, νnGrid, GImp_i, ΣImp_i, μ, dens, fit_quality, fit_res))
                push!(names, (transf_name, opt_name, dist_name))
                i += 1
            end
        end
    end
    return names,fits
end

function GW_fit_real(νnGrid::Vector, p::Vector)::Vector
    tmp = jED.GWeiss_real(νnGrid, μ, p[1:N], p[(N+1):end])
    return tmp
end
names,fits = run_tests(NSites, U, μ, β; maxit = 200, verbose=false)

