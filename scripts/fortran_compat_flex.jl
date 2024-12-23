using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using ForwardDiff
using Optim

# example call: julia scripts/fortran_compat.jl 1.0 1.0 0.5 4 3Dsc-0.20412414523193154 /home/julian/JuliansBastelecke/jED.jl

length(ARGS) < 6 && error("Please proivide U/beta/mu/NBathSites/KGridStr/Path as arguments to the script!")
U = parse(Float64, ARGS[1])
β = parse(Float64, ARGS[2])
μ = parse(Float64, ARGS[3])
NBathSites = parse(Int, ARGS[4])
KGridStr   = ARGS[5]
path       = ARGS[6]
opt_name   = ARGS[7]
cf_type    = ARGS[8]

Nν::Int    = 5000 
maxit::Int = 500
abs_conv::Float64 = 1e-9

function andpar_check_values(ϵₖ, Vₖ)
    NBathSites = length(ϵₖ)
    min_epsk_diff = Inf
    min_Vₖ = minimum(abs.(Vₖ))
    min_eps = minimum(abs.(ϵₖ))
    sum_vk = sum(Vₖ .^ 2)
    for i in 1:NBathSites
        for j in i+1:NBathSites
            if abs(ϵₖ[i] - ϵₖ[j]) < min_epsk_diff
                min_epsk_diff = abs(ϵₖ[i] - ϵₖ[j])
            end
        end
    end
    return sum_vk, min_epsk_diff, min_Vₖ, min_eps
end

"""
    DMFT_Loop(U::Float64, μ::Float64, β::Float64, NBathSites::Int, KGridStr::String; 
              Nk::Int=60, Nν::Int=1000, α::Float64=0.7, abs_conv::Float64=1e-8, ϵ_cut::Float64=1e-15, maxit = 20)

Arguments:
----------
    - U::Float64              : Hubbard U
    - μ::Float64              : chemical potential
    - β::Float64              : inverse temperature
    - NBathSites::Int         : number of bath sites
    - KGridStr::String        : K-Grid String (see Dispersions.jl)
    - Nk::Int=60              : Number of K-Points for GLoc
    - Nν::Int=1000            : Number of Matsubara frequencies
    - α::Float64=0.7          : Mixing (1 -> no mixing)
    - abs_conv::Float64=1e-8  : Difference of Anderson parameters to last iteration for convergence
    - ϵ_cut::Float64=1e-15    : Terms smaller than this are not considered for the impurity Green's function
    - maxit::Int=20           : Maximum number of DMFT iterations

Returns:
----------
    - p       : Anderson parameters
    - νnGrid  : Matsubara grid
    - GImp    : Impurity Green's function
    - ΣImp    : Impurity self-energy
    - dens    : density
"""
function DMFT_Loop(U::Float64, μ::Float64, β::Float64, NBathSites::Int, KGridStr::String, fitf::Function; 
                   Nk::Int=60, Nν::Int=1000, α::Float64=0.8, abs_conv::Float64=1e-8, ϵ_cut::Float64=1e-15, maxit::Int=20)
    ϵₖ = vcat(randn(trunc(Int,NBathSites/2)) .* U .+ U,  randn(trunc(Int,NBathSites/2)) .* U .- U,  repeat([0.2], NBathSites%2))
    Vₖ = randn(NBathSites) ./ 100.0 .+ 0.17
    p  = AIMParams(ϵₖ, Vₖ)
    println(" ======== U = $U / μ = $μ / β = $β / NB = $(length(ϵₖ)) / INIT ======== ")
    println("Solution :    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
    println("              Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing
    dens   = Inf
    Z      = Inf
    done   = false
    i      = 1

    kG     = jED.gen_kGrid(KGridStr, Nk)
    basis  = jED.Basis(length(Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    sum_vk     = Inf
    sum_vk_old = Inf
    min_vk     = Inf
    min_vk_old = Inf
        
    while !done
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis; verbose=false);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=ϵ_cut, overlap=overlap)
        sum_vk_check = abs(sum_vk - 0.25) < (2 * abs(sum_vk_old - 0.25))
        min_vk_check = min_vk > 1e-6 || (min_vk > 0.1 * min_vk_old)
        mix_check    = sum_vk_check && min_vk_check
        !isnothing(GImp_i_old) && mix_check && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        Z = calc_Z(es, β)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        p_old = deepcopy(p)
        fit_res = fitf(p, μ, GLoc_i, νnGrid)
        vals = Optim.converged(fit_res) ? Optim.minimizer(fit_res) : nothing
        isnothing(vals) && (@error("Could not fit, aborting DMFT loop!"); break)
        p = AIMParams(vals[1:NBathSites], vals[NBathSites+1:end])

        sum_vk_old = sum_vk
        min_vk_old = min_vk
        sum_vk, min_eps_diff, min_vk, min_eps = andpar_check_values(vals[1:NBathSites], vals[NBathSites+1:end])
        fit_quality = [sum_vk, min_eps_diff, min_vk, min_eps ]

        tmp = 1 ./ (νnGrid.parent .+ μ .- sum((p.Vₖ .^ 2) ./ (reshape(νnGrid.parent,1,length(νnGrid)) .- p.ϵₖ), dims=1)[1,:])
        err = sum(abs.(GLoc_i.parent .- tmp))
        println(" ======== U = $U / μ = $μ / β = $β / NB = $(length(ϵₖ)) / it = $i ======== ")
        println("   it[$i]: μ = $μ , dens = $dens")
        println("   Converged: ", Optim.converged(fit_res), " // Error = ", err, " // Error after trafo = ", Optim.minimum(fit_res))
        println("   1. min(|Vₖ|)      = ", min_vk)
        println("   2. ∑Vₗ^2          = ", sum_vk)
        println("   3. min(|ϵₖ|)      = ", min_eps)
        println("   4. min(|ϵₖ - ϵₗ|) = ", min_eps_diff)
        println("   Solution :    ϵₖ = $(lpad.(round.(vals[1:NBathSites],digits=4),9)...)")
        println("                 Vₖ = $(lpad.(round.(vals[NBathSites+1:end],digits=4),9)...)")
        println(repeat("=",80))
        if ((sum(abs.(p_old.ϵₖ .- p.ϵₖ)) + sum(abs.(p_old.Vₖ .- p.Vₖ))) < abs_conv) || i >= maxit
            GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=ϵ_cut, overlap=overlap, with_density=true)
            Z = calc_Z(es, β)
            done = true
        end
        i += 1
    end

    return p, νnGrid, GImp_i, ΣImp_i, Z, dens, done
end


# ==================== IO Functions ====================
function write_hubb_andpar(p::AIMParams)
    fname = joinpath(path, "hubb.andpar")
    epsk_str = ""
    tpar_str = ""
    for ek in p.ϵₖ
        epsk_str = epsk_str * "$ek\n"
    end
    for vk in p.Vₖ
        tpar_str = tpar_str * "$vk\n"
    end
    out_string = """           ========================================
              HEADER PLACEHOLDER
           ========================================
NSITE     $NBathSites IWMAX $Nν
    $(β)d0, -12.0, 12.0, 0.007
c ns,imaxmu,deltamu, # iterations, conv.param.
   $NBathSites, 0, 0.d0, $maxit, $abs_conv
c ifix(0,1), <n>,   inew, iauto
Eps(k)
$epsk_str tpar(k)
$tpar_str $μ
"""
    open(fname, "w") do f
        write(f, out_string)
    end
end

function write_νFunction(νnGrid::Vector{ComplexF64}, data::Vector{ComplexF64}, fname::String) 
    row_fmt_str = "%27.16f %27.16f %27.16f"
    row_fmt     = jED.Printf.Format(row_fmt_str * "\n")

    open(fname,"w") do f
        for i in 1:length(νnGrid)
            jED.Printf.format(f, row_fmt, imag(νnGrid[i]), real(data[i]), imag(data[i]))
        end
    end
end

function fitf(pAIM::AIMParams, μ, GLoc_i, νnGrid)
    # THIS CAN BE CHANGED TO POTENTIALLY IMPROVE THE FIT! You should check your data, using example25
    dist_f = jED.square_dist    # |cf(Gw) - cf(G_0)|^2, another option is abs_dist
    # See Optim.jl documentation for a list of optimizers. Least Squares is another option 
    opt = if opt_name == "BFGS" 
        BFGS() 
    elseif opt_name == "cGD"
        ConjugateGradient()
    else
        error("opt_name=$opt_name not found. Please sepcify an optimizer from Optim.jl!  Define preferred optimizer in fortran_compat_flex.jl!")
    end

    cf(x,y) =if cf_type =="inverse_ysqrtx"
             1 ./ (y .* sqrt.(abs.(x)))
    elseif cf_type	=="y"
               y                 # Cost function, here identity. This can be used to emphasize tail, e.g. cf(x, y) = y ./ x
    else
        error("Wrong input for cost function! Define preferred cost function in fortran_compat_flex.jl!")
    end
    opts = Optim.Options(iterations=3000,store_trace = false,
                             show_trace = false,
                             show_warnings = true)
    # END OF FREE PARAMS
    p0 = vcat(pAIM.ϵₖ, pAIM.Vₖ)
    NBathSites = length(pAIM.ϵₖ)
    function wrap_cost(p::Vector)
        GW_i = cf(νnGrid.parent, 1 ./ (νnGrid.parent .+ μ .- sum((p[NBathSites+1:end] .^ 2) ./ (reshape(νnGrid.parent,1,length(νnGrid)) .- p[1:NBathSites]), dims=1)[1,:]))
        GW_i = vcat(real(GW_i),imag(GW_i))
        GL_i = cf(νnGrid.parent, GLoc_i.parent)
        GL_i = vcat(real(GL_i),imag(GL_i))
        return dist_f(GL_i .- GW_i)
    end
    result = optimize(wrap_cost, p0, opt, opts; autodiff = :forward)
    return  result
end

# ==================== Calculation and Output ====================
p, νnGrid, GImp, ΣImp, Z, dens, done = DMFT_Loop(U, μ, β, NBathSites, KGridStr, fitf; abs_conv=abs_conv, Nν=Nν, maxit = maxit)
G0W    = GWeiss(νnGrid, μ, p)

if done
    write_hubb_andpar(p)
    write_νFunction(νnGrid.parent, GImp.parent, joinpath(path, "gm_wim"))  
    write_νFunction(νnGrid.parent, 1 ./ G0W.parent, joinpath(path, "g0m"))  
    open(joinpath(path,"zpart.dat"), "w") do f
        write(f, "$Z")
    end
    open(joinpath(path,"densimp.dat"), "w") do f
        write(f, "$dens")
    end
else
    println("ERROR: DMFT loop did not converge!")
end
