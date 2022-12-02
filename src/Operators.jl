# ==================================================================================================== #
#                                           Operators.jl                                               #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functions for operators and overlaps used in the construction of model Hamiltonians.               #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Refactor overlap_cdagger_c (huge performance bottleneck)                                           #
#   Efficient overlap with 2 creation and 2 annihilation operators, see sfED.jl                        #
# ==================================================================================================== #

# =============================================== Types ==============================================
const OperatorResult = Union{Nothing,Tuple{Int,Fockstate}}

# ============================================= Operators ============================================

"""
    operator_ni(state::Fockstate, i::Int)

Calculate n_i |ket⟩, i.e. `i` is the index for the number operator

Returns: `(eigenval,newState)::Tuple{Int, Fockstate}` with `eigenval` `1` (state `i` occupied) or `0` (state `i` unoccupied)
and `newState`. In case of this operator, `newState === state`.
The density overlap of 2 states can be calculated efficiently using [`overlap_ni_nj`](@ref overlap_ni_nj).

# Example
```
julia> jED.operator_ni(jED.SVector{8}(Bool[1,1,1,1,0,1,0,1]), 2);
(1, Bool[1, 1, 1, 1, 0, 1, 0, 1])
"""
operator_ni(state::Fockstate, i::Int)::OperatorResult = (state[i], state)


# ============================================== Overlaps ============================================
# -------------------------------------------- Fock States -------------------------------------------

"""
    overlap(bra::Fockstate, ket::Fockstate)

Calculates ⟨bra|ket⟩. This is useful when operators have been applied to one state and no efficient direct implementation
for an overlap involving this operator is available.

Returns: True/False
"""
overlap(bra::Fockstate, ket::Fockstate)::Bool = sum(xor.(bra,ket)) == 0

"""
    overlap_cdagger_c(bra::Fockstate, i::Int, ket::Fockstate, j::Int)::Int

Calculate ⟨bra| c^†_i c_j |ket⟩, i.e. `i` is the index for the creation operator
and `j` the index for the annihilation operator.
Internally, we check, that both states have exactly two or no occupation difference.
In both cases the product of [`C_sign`](@ref C_sign) and [`CDag_sign`](@ref CDag_sign) is returned, otherwise 0.
Note, that this allows spin flips!

Returns: -1/0/1

# Example
```
julia> t1 = jED.SVector{8}(Bool[1,1,1,1,0,1,0,1]);
julia> t2 = jED.SVector{8}(Bool[1,1,0,1,1,1,0,1]);
julia> jED.overlap_cdagger_c(t1,t2,3,3)
0
julia> jED.overlap_cdagger_c(t1,t1,2,2)
1
```
"""
function overlap_cdagger_c(bra::Fockstate, createInd::Int, ket::Fockstate, annInd::Int)::Int
    diff = sum(xor.(bra,ket))
    # overlap is only != 0 if exactly 0 or 2 positions don't match
    # we only check sign of the right state, so we manually check of operators on the left state generate 0
    res = if diff == 2 && ((!bra[annInd]) & bra[createInd])
        # This will give 0, when createInd == annInd, even if righ[createInd] == 1. however, in this case the overlap is 0, since diff == 2
        # if annInd < createInd, we need an aditional - sign, since a state annInd is created first
        CDag_sign(ket,createInd)*C_sign(ket,annInd)*(-2*(annInd < createInd) + 1)
    elseif diff == 0 && createInd == annInd
        ket[annInd]
    else
        0
    end
    return res
end

"""
    overlap_ni_nj(bra::Fockstate, ket::Fockstate, i::Int, j::Int)

Calculate ⟨bra| n^†_i n_j |ket⟩

Returns: True/False (converts to 1/0)
"""
function overlap_ni_nj(bra::Fockstate, ket::Fockstate, i::Int, j::Int)::Bool
    return ket[i] & ket[j] & overlap(bra,ket)
end



# ------------------------------------------- Eigen States -------------------------------------------

"""
    overlap_2(bra::Vector, ket::Vector)

Computes |⟨`bra`|`ket`⟩|² for `Float64` or `ComplexF64` type vectors.
"""
function overlap_2(bra::Vector{ComplexF64}, ket::Vector{ComplexF64})::Float64
    c = dot(bra, ket)
    return Float64(conj(c) * c)
end

overlap_2(bra::Vector{Float64}, ket::Vector{Float64})::Float64 = dot(bra,ket)^2
