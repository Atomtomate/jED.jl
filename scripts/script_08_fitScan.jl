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
β  = parse(Float64, ARGS[1])#75.0
kGStr = ARGS[2]
Nk = parse(Int, ARGS[3])
U  = parse(Float64, ARGS[4])
μ  = parse(Float64, ARGS[5])

function andpar_check_values(ϵₖ, Vₖ)
    NSites = length(ϵₖ)
    min_epsk_diff = Inf
    min_Vₖ = minimum(abs.(Vₖ))
    min_eps = minimum(abs.(ϵₖ))
    sum_vk = sum(Vₖ .^ 2)
    for i in 1:NSites
        for j in i+1:NSites
            if abs(ϵₖ[i] - ϵₖ[j]) < min_epsk_diff
                min_epsk_diff = abs(ϵₖ[i] - ϵₖ[j])
            end
        end
    end
    return sum_vk, min_epsk_diff, min_Vₖ, min_eps
end

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
transforms_list  = [transf_01, transf_02, transf_03, transf_04, transf_05, transf_06]
transforms_names = ["y(x) → y(x)", "y(x) → 1/y(x)", "y(x) → y(x)/x", "y(x) → x/y(x)", "y(x) → 1/(y(x)x)", "y(x) → 1/(y(x)√x)"]

optim_list_all  = [NelderMead(), SimulatedAnnealing(), BFGS(), LBFGS(), ConjugateGradient(), GradientDescent(), MomentumGradientDescent(), AcceleratedGradientDescent()]
optim_names_all = ["NelderMead", "SimulatedAnnealing", "BFGS", "LBFGS", "ConjugateGradient", "GradientDescent", "MomentumGradientDescent", "AcceleratedGradientDescent"]
optim_list  = [BFGS(), LBFGS(), ConjugateGradient()]
optim_names = ["BFGS", "LBFGS", "ConjugateGradient"]
opts = Optim.Options(iterations=20000,store_trace = false,
                             show_trace = false,
                             allow_f_increases = false,
                             x_tol = 1e-12,
                             f_tol = 1e-12,
                             show_warnings = true)

dist_list  = [jED.square_dist, jED.abs_dist]
dist_names = ["|vec|^2", "|vec|"]

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


for el in zip(names, fits)
    println(el[1], ": ", el[2][end])
end

for el in zip(names, fits)
    vals = Optim.minimizer(el[2][end])
    println(rpad("=========== $(el[1]) =========",80,"="))
    println(vals)
    println(repeat("=",80))
end
for el in zip(names, fits)
    vals = Optim.minimizer(el[2][end])
    println(rpad("=========== $(el[1]) =========",80,"="))
    println("Converged: ", Optim.converged(el[2][end]), " // Minimum (∑Vₗ^2  = $(sum(vals[NSites+1:end] .^ 2)))")
    println("Solution :    ϵₖ = $(lpad.(round.(vals[1:NSites],digits=4),9)...)")
    println("              Vₖ = $(lpad.(round.(vals[NSites+1:end],digits=4),9)...)")
    println(repeat("=",80))
end


println("Results are available in `fits` variable")
println(to)

jldopen("fit_res_$β.jld2", "w") do f
    fit_quality = map(x->x[end-1], fits)
    f["U"] = U
    f["beta"] = β
    f["NSites"] = NSites
    f["mu"] = μ
    f["kG"] = kGStr
    f["names"] = names
    f["fits"] = fits
    f["fit_quality"] = fit_quality
    f["times"] = to
end
