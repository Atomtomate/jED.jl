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
"""
    AIMParams

Parameters for energy levels (``\\epsilon_k``) and hybridization amplitudes (``V_k``) of the Anderson impurity model.
"""
mutable struct AIMParams
    ϵₖ::Vector{Float64}
    Vₖ::Vector{Float64}
end

function AIMParams(p_vec::Vector)
    length(p_vec) % 2 != 0 && error(
        "p_vec must contain bath energies and hybridizations and therefore have an even length!",
    )
    N = floor(Int, length(p_vec) / 2)
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
struct AIM{NSites,T} <: Model
    tMatrix::SMatrix{NSites,NSites,T}
    UMatrix::SMatrix{NSites,NSites,T}
    JMatrix::SMatrix{NSites,NSites,T}
    params::AIMParams
    μ::Float64
    U::Float64

    AIM(ϵₖ::AbstractVector, Vₖ::AbstractVector, μ::Float64, U::Float64) = AIM{eltype(Vₖ)}(ϵₖ, Vₖ, μ, U)

    function AIM{T}(ϵₖ::AbstractVector, Vₖ::AbstractVector, μ::Float64, U::Float64) where {T}
        length(ϵₖ) != length(Vₖ) && throw(
            ArgumentError(
                "length of ϵₖ $(length(ϵₖ)) must be equal to length of Vₖ $(length(Vₖ))!",
            ),
        )
        NSites = length(ϵₖ) + 1
        tMatrix = collect(Diagonal(cat(T[-μ], ϵₖ, dims=1)))
        tMatrix[2:end, 1] .= Vₖ
        tMatrix[1, 2:end] .= conj(Vₖ)
        UMatrix = zeros(T, NSites, NSites)
        UMatrix[1, 1] = U
        JMatrix = zeros(T, NSites, NSites)
        new{NSites,T}(
            SMatrix{NSites,NSites,T}(tMatrix),
            SMatrix{NSites,NSites,T}(UMatrix),
            SMatrix{NSites,NSites,T}(JMatrix),
            AIMParams(ϵₖ, Vₖ),
            μ,
            U,
        )
    end
end


"""
    Hubbard{NSites,T} <: Model
    Hubbard(t::AbstractMatrix, U::AbstractMatrix)
    Hubbard_Chain(t::Number, U::Number,NSites::Int; pb=false)
    Hubbard_Full(t::Number, U::Number, NSites::Int)

Type for the Hubbard model. A model can be used to construct a [`Eigenspace`](@ref Eigenspace) given
a set of [`Basis`](@ref Basis).

- `pb = true` sets periodic boundary conditions for the hopping matrix in the implicit helper 
constructor that takes numbers instead of the full matrices.

# Example
```
julia> t  = 1.0
julia> U  = 1.0
julia> μ  = 0.5
julia> model = Hubbard(t, U, μ, 3, pb = false)
AIM{3, Float64}([
-0.5 1.0 1.0; 1.0 0.5 0.0; 1.0 0.0 -5.0], [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
```
"""
struct Hubbard{NSites,T} <: Model
    tMatrix::SMatrix{NSites,NSites,T}
    UMatrix::SMatrix{NSites,NSites,T}

    Hubbard(t::AbstractMatrix, U::AbstractMatrix) = Hubbard{eltype(t)}(t, U)

    function Hubbard{T}(tMatrix::AbstractMatrix, UMatrix::AbstractMatrix) where {T}
        NSites = size(tMatrix, 1)
        new{NSites,T}(
            SMatrix{NSites,NSites,T}(tMatrix),
            SMatrix{NSites,NSites,T}(UMatrix)
        )
    end
end

function Hubbard_Chain(t::Number, U::Number, NSites::Int; pb=false)
    tMatrix = diagm(NSites, NSites, 1 => repeat([t], NSites - 1), -1 => repeat([t], NSites - 1))
    if pb
        tMatrix[1,end] = t
        tMatrix[end,1] = t
    end
    UMatrix = diagm(NSites, NSites, 0 => repeat([U], NSites))
    Hubbard{eltype(t)}(tMatrix, UMatrix)
end

function Hubbard_Full(t::Number, U::Number, NSites::Int)
    tMatrix = -t*ones(NSites,NSites) + t*I
    UMatrix = diagm(NSites, NSites, 0 => repeat([U], NSites))
    Hubbard{eltype(t)}(tMatrix, UMatrix)
end


function show(io::IO, ::MIME"text/plain", m::AIM)
    compact = get(io, :compact, false)
    println(
        "Anderson Impurity Model (μ=$(m.μ), U=$(m.U)) with $(size(m.tMatrix,1)) bath sites.",
    )
    println(
        "Bath levels: $(round.(m.params.ϵₖ, digits=2)), Hoppings: $(round.(m.params.Vₖ, digits=2))",
    )
end
