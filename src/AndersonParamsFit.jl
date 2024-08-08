# ==================================================================================================== #
#                                     AndersonParamsFit.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#  Methods for Anderson parameters fitting                                                             #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #

"""
    abs_dist(values::Vector{T})::T

Distance function. Computes ``\\sum_i |values_i |``.
"""
function abs_dist(values::Vector)
    sum(abs.(values))
end

"""
    square_dist(values::Vector{T})::T

Distance function. Computes ``\\sum_i values^2_i``.
"""
function square_dist(values::Vector)
    sum(abs.(values) .^ 2)
end

"""
    cost_01a(nu_data, gLoc_data, gWeiss_data; distf=square_dist)

Cost function for: ``\\mathrm{dist} [ G_{\\mathrm{loc}}(\\nu) - \\mathcal{G}(\\nu)]``

default distance function is ``\\sum_\\nu f^2``, see 
    - [`square_dist`](@ref square_dist)
    - [`abs_dist`](@ref abs_dist)
"""
function cost_01(x_data::Vector, y_data::Vector, yp_data::Vector; distf=square_dist)
    return distf(y_data .- yp_data)
end

"""
    cost_02(nu_data, gLoc_data, gWeiss_data; distf=square_dist)

Cost function for: ``\\mathrm{dist} [ \\frac{1}{\\nu} G_{\\mathrm{loc}}(\\nu) - \\frac{1}{\\nu}\\mathcal{G}(\\nu)]``
"""
function cost_02(x_data::Vector, y_data::Vector, yp_data::Vector; distf=square_dist)
    return distf(y_data ./ x_data .- yp_data ./ x_data)
end

"""
    cost_03(nu_data, gLoc_data, gWeiss_data; distf=square_dist)

Cost function for: `` \\mathrm{dist} [G^{-1}_{\\mathrm{loc}}(\\nu) - \\mathcal{G}^{-1}(\\nu)]``
"""
function cost_03(x_data::Vector, y_data::Vector, yp_data::Vector; distf=square_dist)
    return distf(1 ./ y_data .- 1 ./ yp_data)
end

"""
    cost_04(nu_data, gLoc_data, gWeiss_data; distf=square_dist)

Cost function for: `` \\mathrm{dist} [\\frac{1}{\\nu} G^{-1}_{\\mathrm{loc}}(\\nu) - \\frac{1}{\\nu}\\mathcal{G}^{-1}(\\nu)]``
"""
function cost_04(x_data::Vector, y_data::Vector, yp_data::Vector; distf=square_dist)
    return distf(1 ./  (y_data .* x_data) .- 1 ./  (yp_data .* x_data))
end

"""
    cost_05(nu_data, gLoc_data, gWeiss_data; distf=square_dist)

Cost function for: `` \\frac{1}{N_\\nu} \\sum_\\nu \\frac{1}{\\sqrt{\\nu}} |G^{-1}_{\\mathrm{loc}}(\\nu) - \\mathcal{G}^{-1}(\\nu)|``
"""
function cost_05(x_data::Vector, y_data::Vector, yp_data::Vector; distf=abs_dist)
    return distf(1 ./  (sqrt.(abs.(x_data))) .* (1 ./ y_data .- 1 ./ yp_data)) / (length(x_data)+1)
end

function fit_AIM_params!(
    p::AIMParams,
    GLoc::MatsubaraF,
    μ::Float64,
    νnGrid::FermionicMatsubaraGrid,
)

    tmp = similar(νnGrid.parent)
    p0 = vcat(p.ϵₖ, p.Vₖ)
    N::Int = length(p.ϵₖ)

    function GW_fit_real(νnGrid::Vector, p::Vector)::Vector{Float64}
        GWeiss!(tmp, νnGrid, μ, p[1:N], p[(N+1):end])
        return vcat(real(tmp), imag(tmp))
    end

    target = vcat(real(GLoc.parent), imag(GLoc.parent))
    fit = curve_fit(GW_fit_real, νnGrid.parent, target, p0)
    p.ϵₖ[:] = fit.param[1:N]
    p.Vₖ[:] = fit.param[N+1:end]
end

function model_ED(iν::Vector, p::Vector)
    Δ_fit = zeros(ComplexF64, length(iν))
    for (i, νn) in enumerate(iν)
        tmp = sum((p[(N+1):end] .^ 2) ./ (νn .- p[1:N]))
        Δ_fit[i] = tmp
    end
    return conj.(Δ_fit)
end
