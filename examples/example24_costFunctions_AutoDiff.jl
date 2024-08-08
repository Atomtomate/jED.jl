using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using ForwardDiff
using Optim, LsqFit
using TimerOutputs

to = TimerOutput()

ϵₖ = [1.0, 0.5, -1.1]#, -0.6, 0.05]
Vₖ = [0.25, 0.35, 0.45]#, 0.55, 0.22]
NSites = length(ϵₖ)
p  = AIMParams(ϵₖ, Vₖ)
μ  = 0.6
U  = 1.2
β  = 4.0
tsc= 0.40824829046386307/2
Nν = 300
Nk = 20
α  = 0.2
GImp_i = nothing
GImp_i_old = nothing

kG     = jED.gen_kGrid("3Dsc-$tsc", Nk)
basis  = jED.Basis(length(Vₖ) + 1);
νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
N =   length(ϵₖ)

model  = AIM(ϵₖ, Vₖ, μ, U)
G0W    = GWeiss(νnGrid, μ, p)
es     = Eigenspace(model, basis);
println("     Calculating GImp")
GImp_i, dens = calc_GF_1(basis, es, νnGrid, β)
ΣImp_i = Σ_from_GImp(G0W, GImp_i)
GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)

p0        = vcat(p.ϵₖ, p.Vₖ)
target = vcat(real(GLoc_i.parent), imag(GLoc_i.parent))

transf_01(x, y) = y
transf_02(x, y) = 1 ./ y
transf_03(x, y) = y ./ x
transf_04(x, y) = x ./ y
transf_05(x, y) = 1 ./ (y .* x)
transf_06(x, y) = 1 ./ (y .* sqrt.(abs.(x)))
transforms_list  = [transf_01, transf_02, transf_03, transf_04, transf_05, transf_06]
transforms_names = ["y(x) → y(x)", "y(x) → 1/y(x)", "y(x) → y(x)/x", "y(x) → x/y(x)", "y(x) → 1/(y(x)x)", "y(x) → 1/(y(x)√x)", ]

optim_list  = [BFGS(), LBFGS(), ConjugateGradient(), GradientDescent(), MomentumGradientDescent(), AcceleratedGradientDescent()]
optim_names = ["BFGS", "LBFGS", "ConjugateGradient", "GradientDescent", "MomentumGradientDescent", "AcceleratedGradientDescent"]
opts = Optim.Options(iterations=3000,store_trace = true,
                             show_trace = false,
                             show_warnings = true)

dist_list  = [jED.square_dist, jED.abs_dist]
dist_names = ["|vec|^2", "|vec|"]


function run_tests(νnGrid, GLoc_i)
    fits = []
    names = []
    i = 1
    νnGrid_split = vcat(real(νnGrid.parent),imag(νnGrid.parent))
    global to
    for (transf_name, cf) in zip(transforms_names, transforms_list)
        for (opt_name,opt) in zip(optim_names, optim_list)
            for (dist_name, dist_f) in zip(dist_names, dist_list)
                function wrap_cost(p::Vector)
                    GW_i = cf(νnGrid.parent, 1 ./ (νnGrid.parent .+ μ .- sum((p[NSites+1:end] .^ 2) ./ (reshape(νnGrid.parent,1,length(νnGrid)) .- p[1:NSites]), dims=1)[1,:]))
                    GW_i = vcat(real(GW_i),imag(GW_i))
                    GL_i = cf(νnGrid.parent, GLoc_i.parent)
                    GL_i = vcat(real(GL_i),imag(GL_i))
                    return dist_f(GL_i .- GW_i)
                end
                println("running: ", cf, " // opt: ", typeof(opt))
                @timeit to "run $i" result = optimize(wrap_cost, p0, opt, opts; autodiff = :forward)
                push!(fits, result)
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
names,fits = run_tests(νnGrid, GLoc_i)

@timeit to "LsqFit" lsq_fit = curve_fit(GW_fit_real, νnGrid.parent, target, p0; autodiff=:forwarddiff)

for el in zip(names, fits)
    println(el[1], ": ", el[2])
end

for el in zip(names, fits)
    vals = Optim.minimizer(el[2])
    println(rpad("=========== $(el[1]) =========",80,"="))
    println("Converged: ", Optim.converged(el[2]), " // Minimum (∑Vₗ^2  = $(sum(vals[NSites+1:end] .^ 2)))")
    println("Solution :    ϵₖ = $(lpad.(round.(vals[1:NSites],digits=4),9)...)")
    println("              Vₖ = $(lpad.(round.(vals[NSites+1:end],digits=4),9)...)")
    println(repeat("=",80))
end

println(rpad("=========== Least Squares =========",80,"="))
vals = lsq_fit.param
println("Converged: ", lsq_fit.converged, " // Minimum (∑Vₗ^2  = $(sum(vals[NSites+1:end] .^ 2)))")
println("Solution :    ϵₖ = $(lpad.(round.(vals[1:NSites],digits=4),9)...)")
println("              Vₖ = $(lpad.(round.(vals[NSites+1:end],digits=4),9)...)")
println(repeat("=",80))


println("Results are available in `fits` variable")
println(to)

