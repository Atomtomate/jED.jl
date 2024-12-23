using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED

# example call: julia scripts/fortran_compat.jl 1.0 1.0 0.5 4 3Dsc-0.20412414523193154 /home/julian/JuliansBastelecke/jED.jl

length(ARGS) < 6 && error("Please proivide U/beta/mu/Path/AndersonParameters as arguments to the script!")
U = parse(Float64, ARGS[1])
β = parse(Float64, ARGS[2])
μ = parse(Float64, ARGS[3])
path       = ARGS[4]
andpars    = ARGS[5:end]
NBathSites = floor(Int, length(andpars)/2) 
println("Detected $NBathSites bath sites!")


Nν::Int    = 6000 
abs_conv::Float64 = 1e-9

"""
    DMFT_Fixed(U::Float64, μ::Float64, β::Float64, andpars::Vector; 
              Nν::Int=1000, ϵ_cut::Float64=1e-15)

Arguments:
----------
    - U::Float64              : Hubbard U
    - μ::Float64              : chemical potential
    - β::Float64              : inverse temperature
    - andpars::Vector         : Vector consisting of [BathLevels, Hoppings]
    - Nν::Int=1000            : Number of Matsubara frequencies
    - ϵ_cut::Float64=1e-15    : Terms smaller than this are not considered for the impurity Green's function

Returns:
----------
    - p       : Anderson parameters
    - νnGrid  : Matsubara grid
    - G0W     : Weiss Green's functions
    - GImp    : Impurity Green's function
    - ΣImp    : Impurity self-energy
    - dens    : density
"""
function DMFT_Fixed(U::Float64, μ::Float64, β::Float64, andpars::Vector; 
                   Nν::Int=1000, ϵ_cut::Float64=1e-15)
    NBathSites = floor(Int, length(andpars)/2) 
    println(andpars)
    println(length(andpars))
    ϵₖ = parse.(Float64, andpars[1:NBathSites])
    Vₖ = parse.(Float64, andpars[NBathSites+1:end])
    p  = AIMParams(ϵₖ, Vₖ)
    println(" ======== U = $U / μ = $μ / β = $β / NB = $(length(ϵₖ)) / INIT ======== ")
    println("Anderson Parameters:   ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
    println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")

    basis  = jED.Basis(length(Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
    model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
    G0W    = GWeiss(νnGrid, μ, p)
    es     = Eigenspace(model, basis);
    GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=ϵ_cut, overlap=overlap, with_density=true)
    ΣImp_i = Σ_from_GImp(G0W, GImp_i)
    Z = calc_Z(es, β)

    return p, νnGrid, G0W, GImp_i, ΣImp_i, Z, dens
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
   NSITE     $(NBathSites+1) IWMAX  $Nν
    $(β)d0, -12.0, 12.0, 0.007
c ns,imaxmu,deltamu, # iterations, conv.param.
    $(NBathSites+1), 0, 0.d0, 1, $abs_conv
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

# ==================== Calculation and Output ====================
p, νnGrid, G0W, GImp, ΣImp, Z, dens = DMFT_Fixed(U, μ, β, andpars, Nν=Nν)

write_hubb_andpar(p)
write_νFunction(νnGrid.parent, GImp.parent, joinpath(path, "gm_wim"))  
write_νFunction(νnGrid.parent, 1 ./ G0W.parent, joinpath(path, "g0m"))  
open(joinpath(path,"zpart.dat"), "w") do f
    write(f, "$Z")
end
open(joinpath(path,"densimp.dat"), "w") do f
    write(f, "$dens")
end
