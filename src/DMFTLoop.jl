# ==================================================================================================== #
#                                          DMFTLoop.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#  Stub for DMFT Loop and Anderson parameter fitting                                                   #
# -------------------------------------------- TODO -------------------------------------------------- #
#   This is only a stub, needs to be properly wrapped in types
# ==================================================================================================== #

const FermionicMatsubaraGrid = OffsetVector{ComplexF64, Vector{ComplexF64}}
const MatsubaraF = OffsetVector{ComplexF64, Vector{ComplexF64}}

"""
    Σ_from_GImp(GWeiss::OffsetVector{ComplexF64, Vector{ComplexF64}}, GImp::OffsetVector{ComplexF64, Vector{ComplexF64}})

Computes self-energy from impurity Green's function (as obtained from a given impurity solver) and the [`Weiss Greens Function`](@ref GWeiss_from_Δ).
"""
function Σ_from_GImp(GWeiss::MatsubaraF, GImp::MatsubaraF)
    return 1 ./ GWeiss .- 1 ./ GImp
end


"""
    GWeiss_from_Δ(Δ::OffsetVector{ComplexF64, Vector{ComplexF64}}, μ::Float64, νnGrid::FermionicMatsubaraGrid)

Computes Weiss Green's frunction from [`hybridization function`](@ref Δ_AIM).
"""
function GWeiss(Δ::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid)
    return 1 ./ (νnGrid .+ μ .- Δ)
end

"""
    GWeiss(νnGrid::FermionicMatsubaraGrid, p::AIMParams)
    GWeiss!(target::Vector, νnGrid::Vector, μ::Float64, p::AIMParams)

Computes Weiss Green's frunction from [`Anderson Parameters`](@ref AIMParams).
"""
function GWeiss(νnGrid::FermionicMatsubaraGrid, μ::Float64, p::AIMParams)
    res = similar(νnGrid.parent)
    GWeiss!(res, νnGrid.parent, μ, p.ϵₖ, p.Vₖ)
    return OffsetVector(res, axes(νnGrid))
end

function GWeiss!(target::Vector, νnGrid::Vector, μ::Float64, ϵₖ::Vector, Vₖ::Vector)
    length(target) != length(νnGrid) && error("νnGrid and target must have the same length!")
    for i in 1:length(νnGrid)
        target[i] = 1 / (νnGrid[i] + μ - sum((Vₖ .^ 2) ./ (νnGrid[i] .- ϵₖ)))
    end
end

"""
    GWeiss_from_Imp(GLoc::MatsubaraF, ΣImp::MatsubaraF)

Compute Updates Weiss Green's function from impurity self-energy via ``[G_\\text{loc} + \\Sigma_\\text{Imp}]^{-1}`` (see [`GLoc`](@ref GLoc) and [`Σ_from_GImp`](@ref Σ_from_GImp)).
"""
function GWeiss_from_Imp(GLoc::MatsubaraF, ΣImp::MatsubaraF)
    return 1 ./ (ΣImp .+ 1 ./ GLoc)
end

"""
    Δ_AIM(νnGrid::FermionicMatsubaraGrid, p::AIMParams)
    Δ_AIM(νnGrid::FermionicMatsubaraGrid, p::Vector{Float64})

Computes hybridization function ``\\sum_p \\frac{V_p^2}{i\\nu_n - \\epsilon_p}`` from [`Anderson Impurity Model Parameters`](@ref AIMParams) with ``p`` bath sites.
"""
function Δ_AIM(νnGrid::FermionicMatsubaraGrid, p::AIMParams)
    return OffsetVector(Δ_AIM(νnGrid.parent, vcat(p.ϵₖ, p.Vₖ)),eachindex(νnGrid))
end

function Δ_AIM(νnGrid::Vector{ComplexF64}, p::Vector{Float64})
    Δ = similar(νnGrid)
    N::Int = floor(Int, length(p)/2)
    Δint(νn::ComplexF64, p::Vector{Float64})::ComplexF64 = conj(sum(p[(N+1):end] .^ 2 ./ (νn .- p[1:N])))
    for νi in eachindex(νnGrid)
        Δ[νi] = Δint(νnGrid[νi], p)
    end
    return Δ
end

function Δ_AIM_real(νnGrid::Vector{ComplexF64}, p::Vector{Float64})
    Δ = similar(νnGrid)
    N::Int = floor(Int, length(p)/2)
    Δint(νn::ComplexF64, p::Vector{Float64})::ComplexF64 = conj(sum(p[(N+1):end] .^ 2 ./ (νn .- p[1:N])))
    for νi in eachindex(νnGrid)
        Δ[νi] = Δint(νnGrid[νi], p)
    end
    return vcat(real.(Δ), imag.(Δ))
end


"""
    Δ_from_GWeiss(GWeiss::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid)

Computes hybridization function from Weiss Green's function via ``\\Delta^{\\nu} = i\\nu_n + \\mu - \\left(\\mathcal{G}^{\\nu}_0\\right^{-1}``.
"""
function Δ_from_GWeiss(GWeiss::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid)
    return νnGrid .- μ .- 1 ./ GWeiss
end

"""
    GLoc(ΣImp::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid, kG::KGrid)

Compute local Green's function ``\\int dk [i\\nu_n + \\mu - \\epsilon_k - \\Sigma_\\text{Imp}(i\\nu_n)]^{-1}``.
TODO: simplify -conj!!!
"""
function GLoc(ΣImp::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid, kG::KGrid)
    GLoc = similar(ΣImp)
    tmp = μ .- dispersion(kG)
    for νi in eachindex(νnGrid)
        νn = νnGrid[νi]
        GLoc[νi] = kintegrate(kG, 1 ./ (νn .+ tmp .- ΣImp[νi]))
    end
    GLoc = 1 ./ (1 ./ GLoc .+ ΣImp)
    return GLoc
end

function fit_AIM_params!(p::AIMParams, GLoc::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid)

    tmp = similar(νnGrid.parent)
    p0  = vcat(p.ϵₖ, p.Vₖ)
    N::Int = floor(Int, length(p0)/2)

    function GW_fit_real(νnGrid::Vector, p::Vector)::Vector{Float64} 
        GWeiss!(tmp, νnGrid, μ, p[1:N], p[(N+1):end])
        vcat(real(tmp), imag(tmp))
    end

    target = vcat(real(GLoc.parent),imag(GLoc.parent))
    fit = curve_fit(GW_fit_real, νnGrid.parent, target, p0)
    p.ϵₖ[:] = fit.param[1:N]
    p.Vₖ[:] = fit.param[N+1:end]
end

function model_ED(iν, p)
    Δ_fit = zeros(ComplexF64,length(iν))
    for (i,νn) in enumerate(iν)
        tmp  = sum((p[(N+1):end] .^ 2) ./ (νn .- p[1:N]))
        Δ_fit[i] = tmp
    end
    return conj.(Δ_fit)
end