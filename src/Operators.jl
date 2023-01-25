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

"""
    Operator

Operator for [`Fockstates`](@ref Fockstate). Holds function for transformation of Fock state and change in quantum numbers.

Fields
-------------
- **`f`**        : Function, operation on Fock state
- **`N_inc`**    : Integer, change in electron number
- **`S_inc`**    : Integer, change in spin number

# Example
julia> o1 = Operator(x->create())
"""
struct Operator
    f::Function
    N_inc::Int
    S_inc::Int
end

(op::Operator)(s::Fockstate) = op.f(s)
    

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
```
"""
operator_ni(state::Fockstate, i::Int)::OperatorResult = (state[i], state)

"""
    create(state::Fockstate{Length}, i::Int)::Union{Nothing,Fockstate} where Length

Returns either a new [`Fockstate`](@ref Fockstate) with an electron `i` created (see layout of Fockstate, how up and down is encoded),
or `nothing`, if `i` is already occupied. See also [`ann`](@ref ann).
See also [`create_op`](@ref create_op) for the [`Operator`](@ref Operator) version.
"""
function create(state::Fockstate{Length}, i::Int)::Union{Nothing,Fockstate} where Length
    if state[i]
        return nothing
    else
        return Fockstate{Length}([s != i ? state[s] : 1 for s in 1:Length])
    end
end

"""
    create_op(b::Basis, i::Int)

[`Operator`](@ref Operator) for creation of state `i` (for basis length N, i > N creates spin down, otherwise spin up).
"""
function create_op(b::Basis, i::Int)
    len = typeof(b).parameters[1]
    S_inc = 2*(i < len/2) - 1
    Operator(s->create(s,i), 1, S_inc)
end

"""
    ann(state::Fockstate{Length}, i::Int)::Union{Nothing,Fockstate} where Length

Returns either a new [`Fockstate`](@ref Fockstate) with an electron `i` annihilated (see layout of Fockstate, how up and down is encoded),
or `nothing`, if `i` is not occupied. See also [`create`](@ref create).
See also [`ann_op`](@ref ann_op) for the [`Operator`](@ref Operator) version.
"""
function ann(state::Fockstate{Length}, i::Int)::Union{Nothing,Fockstate} where Length
    if !state[i]
        return nothing
    else
        return Fockstate{Length}([s != i ? state[s] : 0 for s in 1:Length])
    end
end

"""
    anne_op(b::Basis, i::Int)

[`Operator`](@ref Operator) for annihilation of state `i` (for basis length N, i > N annihilates spin down, otherwise spin up).
"""
function ann_op(b::Basis, i::Int)
    len = typeof(b).parameters[1]
    S_inc = 2*(i > len/2) - 1
    Operator(s->ann(s,i), -1, S_inc)
end

# ============================================== Overlaps ============================================
# -------------------------------------------- Fock States -------------------------------------------

"""
    overlap(bra::Fockstate, ket::Fockstate)

Calculates ⟨`bra`|`ket`⟩. This is useful when operators have been applied to one state and no efficient direct implementation
for an overlap involving this operator is available.

Returns: True/False
"""
overlap(bra::Fockstate, ket::Fockstate)::Bool = sum(xor.(bra,ket)) == 0

"""
    overlap_cdagger_c(bra::Fockstate, i::Int, ket::Fockstate, j::Int)::Int

Calculates ``\\langle`` `bra` ``| c^\\dagger_i c_j | `` `ket` ``\\rangle``, i.e. `i` is the index for the creation operator
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


"""
    _find_cdag_overlap_blocks(blocklist::Vector{Blockinfo}; S_inc=-1, N_inc = 1)

Find block index with [`N_el`](@ref N_el) increased by `1` and [`S`](@ref S) either in increased or decreased (depending on `S_inc`), e.g. block index of state ``\\langle j |``
with non-zero overlap of ``c^\\dagger_\\uparrow |i \\rangle``.

Returns: Vector{Int}, equal length to `blocklist`. Each entry with index `i` contains the index `j` of the block
with one more electron and spin (i.e. the block for which ``\\langle j| c^\\dagger | i \\rangle`` does NOT vanish).
This is stored here for performance reasons, since these overlaps are used often in the Lehman representation.
TODO: do not hardcode spin up GF!!! (this forces S+1 instead of S+-1 for now)
"""
function _find_cdag_overlap_blocks(blocklist::Vector{Blockinfo}, op::Operator)
    res = Array{Int}(undef, length(blocklist))
    for bi in 1:length(blocklist)
        _, _, Nel_ket, S_ket = blocklist[bi]
        indN = searchsorted(map(x->x[3],blocklist), Nel_ket+op.N_inc)
        if length(indN) > 0
            indS = searchsortedfirst(map(x->x[4],blocklist[indN]), S_ket+op.S_inc)
            val = first(indN) + indS - 1
            res[bi] = val <= length(blocklist) && blocklist[val][4] == S_ket+op.S_inc ? val : -1
        else
            res[bi] = -1
        end
    end
    return res
end

"""
_overlap_list(basis::Basis[, i::Int, j::Int], op::Function)

Calculates the overlap between block `i` and `j` under `op`. Builds full overlap indices for all block if `i` and `j` are not provided.

# Example
```
julia> basis = jED.Basis(3)
julia> op(b) = jED.create(b, 4)  # creation operator for spin down at site 1 of 3
julia> jED._overlap_list(basis, 2, 4, op)
[0, 1, 2]
``` 
"""
function _overlap_list(basis::Basis, i::Int, j::Int, op::Function)
    slice_i = _block_slice(basis.blocklist[i])
    slice_j = _block_slice(basis.blocklist[j])
    res = zeros(Int, length(slice_i))
    for (i,bi) in enumerate(basis.states[slice_i])
        b_new = op(bi)
        for (j,bj) in enumerate(basis.states[slice_j])
            all(b_new.== bj) && (res[i] = j)
        end
    end
    return res
end

function _overlap_list(basis::Basis, op::Function)
    ov_i = _find_cdag_overlap_blocks(basis.blocklist, op)
    res = zeros(Int, length(basis.states))
    for (i,j) in enumerate(ov_i)
        slice_i = _block_slice(basis.blocklist[i])
        if j != -1
            res[slice_i] = _overlap_list(basis, i, j, op)
        end
    end
    return res
end
# function _build_cdag_overlap_slices(blocklist::Vector{Blockinfo})
#     overlap_target = _find_cdag_overlap_blocks(blocklist)
#     res = Vector{Tuple{UnitRange{Int},UnitRange{Int}}}(undef, count(overlap_target .> 0))
#     j = 1
#     for i in 1:length(blocklist)
#         if overlap_target[i] > 0
#             res[j] = (_block_slice(blocklist[i]), _block_slice(blocklist[overlap_target[i]]))
#             j += 1
#         end
#     end
#     return res
# end
