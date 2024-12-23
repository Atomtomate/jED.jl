using Pkg
Pkg.activate("/scratch/projects/hhp00048/codes/jED.jl")
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
dens_goal  = parse(Float64,ARGS[7])

Nν::Int    = 5000 
maxit::Int =1000 
abs_conv::Float64 = 1e-8
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
function DMFT_Loop(U::Float64, μ_in::Float64, β::Float64, NBathSites::Int, KGridStr::String, fitf::Function ,dens_fix::Float64; 
                   Nk::Int=60, Nν::Int=1000, α::Float64=0.7, abs_conv::Float64=1e-8, ϵ_cut::Float64=1e-15, maxit::Int=20)
    ϵₖ = [iseven(NBathSites) || i != ceil(Int, NBathSites/2) ? (U/2)/(i-NBathSites/2-1/2) : 0 for i in 1:NBathSites]
    Vₖ = [1/(4*NBathSites) for i in 1:NBathSites]
    p  = AIMParams(ϵₖ, Vₖ)
    μ      = μ_in
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
        
    while !done
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=ϵ_cut, overlap=overlap)
        Nup = calc_Nup(es, β, basis, model.impuritySiteIndex)
        Ndo = calc_Ndo(es, β, basis, model.impuritySiteIndex)
        dens = Nup + Ndo
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        Z = calc_Z(es, β)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        p_old = deepcopy(p)
        fit_res = fitf(p, μ, GLoc_i, νnGrid)
        vals = Optim.converged(fit_res) ? Optim.minimizer(fit_res) : nothing
        isnothing(vals) && (@error("Could not fit, aborting DMFT loop!"); break)
        p = AIMParams(vals[1:NBathSites], vals[NBathSites+1:end])
        println(" ======== U = $U / μ = $μ / β = $β / NB = $(length(ϵₖ)) / it = $i ======== ")
        println("Solution :    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("              Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2)) // Z = $Z")
        println("μ = $μ , dens = $dens")

        if i > 5
            μ = μ - (dens - dens_fix)/10
        end
        println("Convergence parameter: " * string(sum(abs.(p_old.ϵₖ .- p.ϵₖ)) + sum(abs.(p_old.Vₖ .- p.Vₖ))))      
        if ((sum(abs.(p_old.ϵₖ .- p.ϵₖ)) + sum(abs.(p_old.Vₖ .- p.Vₖ))) < abs_conv) || i >= maxit
            GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=ϵ_cut, overlap=overlap, with_density=true)
            Z = calc_Z(es, β)
            done = true
        end
        i += 1
    end

    return p, νnGrid, GImp_i, ΣImp_i, Z, dens, μ
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
    opt = LBFGS()                # See Optim.jl documentation for a list of optimizers. Least Squares is another option 
    cf(x,y) =  x ./ y                 # Cost function, here identity. This can be used to emphasize tail, e.g. cf(x, y) = y ./ x
    opts = Optim.Options(iterations=3000,store_trace = false,
                             show_trace = false,
                             show_warnings = true)
    # END OF FREE PARAMS
    p0 = vcat(pAIM.ϵₖ, pAIM.Vₖ)
    NSites = length(pAIM.ϵₖ)
    function wrap_cost(p::Vector)
        GW_i = cf(νnGrid.parent, 1 ./ (νnGrid.parent .+ μ .- sum((p[NSites+1:end] .^ 2) ./ (reshape(νnGrid.parent,1,length(νnGrid)) .- p[1:NSites]), dims=1)[1,:]))
        GW_i = vcat(real(GW_i),imag(GW_i))
        GL_i = cf(νnGrid.parent, GLoc_i.parent)
        GL_i = vcat(real(GL_i),imag(GL_i))
        return dist_f(GL_i .- GW_i)
    end
    result = optimize(wrap_cost, p0, opt, opts; autodiff = :forward)
    return  result
end

# ==================== Calculation and Output ====================
p, νnGrid, GImp, ΣImp, Z, dens, μ = DMFT_Loop(U, μ, β, NBathSites, KGridStr, fitf, dens_goal; abs_conv=abs_conv, Nν=Nν, maxit = maxit)

G0W    = GWeiss(νnGrid, μ, p)

write_hubb_andpar(p)
write_νFunction(νnGrid.parent, GImp.parent, joinpath(path, "gm_wim"))  
write_νFunction(νnGrid.parent, 1 ./ G0W.parent, joinpath(path, "g0m"))  
open(joinpath(path,"zpart.dat"), "w") do f
    write(f, "$Z")
end
open(joinpath(path,"densimp.dat"), "w") do f
    write(f, "$dens")
end
