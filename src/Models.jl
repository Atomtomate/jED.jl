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

# ===================== Anderson Impurity Model =====================
mutable struct AIMParams
    ϵₖ::Vector{Float64}
    Vₖ::Vector{Float64}
end

function AIMParams(p_vec::Vector)
    length(p_vec) % 2 != 0 && error("p_vec must contain bath energies and hybridizations and therefore have an even length!")
    N = floor(Int,length(p_vec)/2)
    AIMParams(p_vec[1:N], p_vec[N+1:end])
end

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
```
"""
struct AIM{NSites, T} <: Model
    tMatrix::SMatrix{NSites,NSites,T}
    UMatrix::SMatrix{NSites,NSites,T}
    JMatrix::SMatrix{NSites,NSites,T}
    params::AIMParams
    μ::Float64
    U::Float64
    function AIM(ϵₖ::Vector{Float64}, Vₖ::Vector{T}, μ::Float64, U::Float64) where T <: Real
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
                           SMatrix{NSites,NSites,T}(JMatrix),
                           AIMParams(ϵₖ, Vₖ), μ, U)
    end
end


function show(io::IO, ::MIME"text/plain", m::AIM)
    compact = get(io, :compact, false)
    println("Anderson Impurity Model (μ=$(m.μ), U=$(m.U)) with $(size(m.tMatrix,1)) bath sites.")
    println("Bath levels: $(round.(m.params.ϵₖ, digits=2)), Hoppings: $(round.(m.params.Vₖ, digits=2))")
end
