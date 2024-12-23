using Pidfile
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using Optim, LsqFit


U     = parse(Float64, ARGS[1])
β     = parse(Float64, ARGS[2])
NSites= parse(Int, ARGS[3])
μin   = parse(Float64,ARGS[4])
μc    = parse(Float64, ARGS[5])
μfi   = parse(Float64, ARGS[6])
out_path  = ARGS[7]
ID  = ARGS[8]


ek_start = if NSites == 4 && μin < μfi
        [-0.12856018041109396, 0.9190656650224953, 0.09715022120548436, -1.0987087036542595]
    elseif NSites ==4 && μin>μfi
        [-0.3409956086473959, 1.4041543311998974, 0.49412637746901317, -1.272211814539253]
        #[0.0298, 0.2937,-0.1117,-0.5875] 
    elseif NSites == 5 
        [0.0291,  0.2296, -0.1289, -0.4249,  0.6445]
    elseif NSites == 6
        [0.0291,  0.2296, -0.1289, -0.4249,  0.6445,  -0.887]
    end
vk_start = if NSites == 4 
        [0.09026224888735168, 0.3158803717931892, 0.09170871169958564, 0.36144662824304025]
        #[0.1384, 0.2088, 0.2032, 0.3954] 
    elseif NSites == 5 
        [0.153, 0.1794,  0.1985,  0.2763,  0.1635]
    elseif NSites == 6
        [0.153, 0.1794,  0.1985,  0.2763,  0.1635,  0.2902]
end

function DMFT_Loop(U::Float64, μ_in::Float64, β::Float64, dens_fix::Float64, p; maxit = 20)
    Nν  = 2000
    Nk  = 500
    α   = 0.4
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing

    kG     = jED.gen_kGrid("2Dsc-0.25-0.075-0.05",Nk)
    basis  = jED.Basis(length(p.Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    μ = μ_in
    dens = NaN
    model  = nothing
    es = nothing
    fit_res = nothing
        
    for i in 1:maxit
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis, verbose=false);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-14, overlap=overlap, with_density=false)
        Nup = calc_Nup(es, β, basis, model.impuritySiteIndex)
        Ndo = calc_Ndo(es, β, basis, model.impuritySiteIndex)
        dens = Nup + Ndo
        (i < 10) && !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        !isnothing(GImp_i_old) && sum(abs.(GImp_i_old .- GImp_i)) < 1e-8 && break
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        #
        fit_res = fitf(p, μ, GLoc_i, νnGrid)
        vals = Optim.converged(fit_res) ? Optim.minimizer(fit_res) : nothing
        isnothing(vals) && (@warn("Could not fit, aborting DMFT loop!"); break)
        p = AIMParams(vals[1:NSites], vals[NSites+1:end])
        sum_vk, min_eps_diff, min_vk, min_eps = andpar_check_values(vals[1:NSites], vals[NSites+1:end])

        tmp = 1 ./ (νnGrid.parent .+ μ .- sum((p.Vₖ .^ 2) ./ (reshape(νnGrid.parent,1,length(νnGrid)) .- p.ϵₖ), dims=1)[1,:])
        err = sum(abs.(GLoc_i.parent .- tmp))
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

        #fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        #println("Solution using Lsq:   ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        #println("                      Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        #println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
        #println("μ = $μ , dens = $dens")
        #
#        if i > 5
#            μ = μ - (dens - dens_fix)/10
#        end
    end
    D   = jED.calc_D(es, β, basis, model.impuritySiteIndex)
    return p, νnGrid, GImp_i, ΣImp_i,μ, dens, D
end


opts = Optim.Options(iterations=5000,store_trace = true,
                             show_trace = false,
                             show_warnings = true)




transf_name="y(x) → y(x)" 
#transf_01(x, y) = y
#cf=transf_01(x, y)
dist_name="|vec|^2"
dist_f=jED.square_dist
optim_name = "BFGS"
opt=ConjugateGradient()



function fitf(pAIM::AIMParams, μ, GLoc_i, νnGrid)
    p0 = vcat(pAIM.ϵₖ, pAIM.Vₖ)
    function wrap_cost(p::Vector)
        GW_i = 1 ./ (νnGrid.parent .+ μ .- sum((p[NSites+1:end] .^ 2) ./ (reshape(νnGrid.parent,1,length(νnGrid)) .- p[1:NSites]), dims=1)[1,:])
        GW_i = vcat(real(GW_i),imag(GW_i))
        GL_i = GLoc_i.parent
        GL_i = vcat(real(GL_i),imag(GL_i))
        return dist_f(GL_i .- GW_i)
    end
    #println("running: ", cf, " // opt: ", typeof(opt))
    #@timeit to "run $i" 
    result = optimize(wrap_cost, p0, opt, opts; autodiff = :forward)
    return  result
end



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



#p = AIMParams(ek_start, vk_start)
#
#p, νnGrid, GImp, ΣImp, μ, dens, D = DMFT_Loop(U, μ, β, 1.0, p, maxit = 1000);
#mkpidlock(joinpath(out_path,"lock_$ID.pid")) do
#   open(joinpath(out_path,"results_$ID.txt"), lock=true, append=true) do f
#      println(f, "$U,$β,$μ,$dens,$D,$(p.ϵₖ),$(p.Vₖ)")
#   end
#end   



function run_scan()
    p = AIMParams(ek_start, vk_start)
    μr = if  μin<μfi
        vcat(μin:+0.01:μc-0.01001,μc-0.01:+0.001:μc+0.00901,μc+0.01:+0.01:μfi+0.00001)
    else
        vcat(μin:-0.01:μc+0.01001,μc+0.01:-0.001:μc-0.00901,μc-0.01:-0.01:μfi-0.00001)  
    end
    # Ur = from0 ? LinRange(0,6,26)[7:20] : reverse(LinRange(0,6,26)[7:20])
    for μi in μr
        p, νnGrid, GImp, ΣImp, μ, dens, D = DMFT_Loop(U, μi, β, 1.0, p, maxit = 1000);
        mkpidlock(joinpath(out_path,"lock_$ID.pid")) do
            open(joinpath(out_path,"results_$ID.txt"), lock=true, append=true) do f
                println(f, "$μi,$β,$U,$dens,$D,$(p.ϵₖ),$(p.Vₖ)")
            end
        end
    end
end
run_scan()
