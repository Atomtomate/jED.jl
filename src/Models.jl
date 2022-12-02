# ==================================================================================================== #
#                                            Models.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Hamiltonians for different models.                                                                 #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #

"""
    Model

Abstract type for model type. 
Currently implemented:
  - [`AIM`](@ref AIM)
"""
abstract type Model end

"""
    AIM <: Model
    AIM(ϵₖ::Vector{Float64}, Vₖ::Vector{T}, μ::Float64, U::Float64)

Type for the Anderson impurity model. A model can be used to construct a [`Eigenspace`](@ref Eigenspace) given
a set of [`Basis`](@ref Basis).

Can be created from 
  - bath onsite energies ``\\epsilon_k``,
  - bath-impurity hopping ``V_k``
  - impurity chemical potential ``\\mu``
  - Coulomb interaction strength on the impurity ``U``

# Example
```
julia> ϵₖ = [0.5, -5.0]
julia> Vₖ = [1.0, 1.0]
julia> U  = 1.0
julia> μ  = 0.5
julia> model = AIM(ϵₖ, Vₖ, μ, U)
AIM{3, Float64}([
-0.5 1.0 1.0; 1.0 0.5 0.0; 1.0 0.0 -5.0], [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
"""
struct AIM{NSites, T} <: Model
    tMatrix::SMatrix{NSites,NSites,T}
    UMatrix::SMatrix{NSites,NSites,T}
    JMatrix::SMatrix{NSites,NSites,T}
    function AIM(ϵₖ::Vector{Float64}, Vₖ::Vector{T}, μ::Float64, U::Float64) where T <: Union{ComplexF64, Float64}
        length(ϵₖ) != length(Vₖ) && throw(ArgumentError("length of ϵₖ $(length(ϵₖ)) must be equal to length of Vₖ $(length(Vₖ))!"))
            NSites = length(ϵₖ) + 1
            tMatrix = collect(Diagonal(cat(T[-μ], ϵₖ ,dims=1)))
            tMatrix[2:end,1] .= Vₖ
            tMatrix[1,2:end] .= conj(Vₖ)
            UMatrix = zeros(T, NSites, NSites)
            UMatrix[1,1] = U
            JMatrix = zeros(T, NSites, NSites)
            new{NSites, T}(SMatrix{NSites,NSites,T}(tMatrix), 
                           SMatrix{NSites,NSites,T}(UMatrix), 
                           SMatrix{NSites,NSites,T}(JMatrix))
    end
end
