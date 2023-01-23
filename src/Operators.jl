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
```
"""
operator_ni(state::Fockstate, i::Int)::OperatorResult = (state[i], state)

"""
    create(state::Fockstate{Length}, i::Int)::Union{Nothing,Fockstate} where Length

Returns either a new [`Fockstate`](@ref Fockstate) with an electron `i` created (see layout of Fockstate, how up and down is encoded),
or `nothing`, if `i` is already occupied. See also [`ann`](@ref ann).
"""
function create(state::Fockstate{Length}, i::Int)::Union{Nothing,Fockstate} where Length
    if state[i]
        return nothing
    else
        return Fockstate{Length}([s != i ? state[s] : 1 for s in 1:Length])
    end
end

"""
    ann(state::Fockstate{Length}, i::Int)::Union{Nothing,Fockstate} where Length

Returns either a new [`Fockstate`](@ref Fockstate) with an electron `i` annihilated (see layout of Fockstate, how up and down is encoded),
or `nothing`, if `i` is not occupied. See also [`create`](@ref create).
"""
function ann(state::Fockstate{Length}, i::Int)::Union{Nothing,Fockstate} where Length
    if !state[i]
        return nothing
    else
        return Fockstate{Length}([s != i ? state[s] : 0 for s in 1:Length])
    end
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
    overlap_cdagger_2(fockbasis::Basis, ci::Int, bra::Vector, ket)
"""
function _find_cdag_overlap_index(basis::Basis, block_overlap, site::Int)
    site != 1 && error("site != 1 not implemented yet (forces measurement of spin up GF)!")
    res = zeros(Int,length(basis.states))
    i::Int = 1
    for (ket_slice,bra_slice) in block_overlap
        for ket_i in basis.states[ket_slice] 
            cdag_ket = create(ket_i, site)
            res[i] = if cdag_ket === nothing
                0
            else
                overlap_index = findfirst(x->x == cdag_ket, basis.states[bra_slice])
                overlap_index === nothing ? 0 : overlap_index
            end
            i += 1
        end
    end
    return res
end

function _find_cdag_overlap_index_naive(basis::Basis, site::Int)
    res = zeros(Int,length(basis.states))
    i::Int = 1
    for block_el in basis.blocklist
        ket_slice = _block_slice(block_el)
        for ket_i in basis.states[ket_slice] 
            cdag_ket = create(ket_i, site)
            tmp = if cdag_ket !== nothing
                overlap_index = findfirst(x->x == cdag_ket, basis.states)
                if overlap_index !== nothing
                    block_i = findfirst(x-> x[1] > overlap_index,basis.blocklist)
                    println(i,",",overlap_index,",",block_i)
                    if block_i !== nothing
                        println(overlap_index, basis.blocklist[block_i-1])
                        overlap_index - basis.blocklist[block_i-1][1] + 1
                    end
                end
            end
            res[i] = tmp !== nothing ? tmp : 0
            i += 1
        end
    end
    return res
end


"""
    _find_cdag_overlap_blocks(blocklist::Vector{Blockinfo}; insert_spin=-1)

Find block index with [`N_el`](@ref N_el) increased by `1` and [`S`](@ref S) either in increased or decreased (depending on `insert_spin = 1` or `-1`), i.e. block index of state ``\\langle j |``
with non-zero overlap of ``c^\\dagger_\\uparrow |i \\rangle``.

Returns: Vector{Int}, equal length to `blocklist`. Each entry with index `i` contains the index `j` of the block
with one more electron and spin (i.e. the block for which ``\\langle j| c^\\dagger | i \\rangle`` does NOT vanish).
This is stored here for performance reasons, since these overlaps are used often in the Lehman representation.
TODO: do not hardcode spin up GF!!! (this forces S+1 instead of S+-1 for now)
"""
function _find_cdag_overlap_blocks(blocklist::Vector{Blockinfo}; insert_spin=-1)
    res = Array{Int}(undef, length(blocklist))
    for bi in 1:length(blocklist)
        _, _, Nel_ket, S_ket = blocklist[bi]
        indN = searchsorted(map(x->x[3],blocklist), Nel_ket+1)
        if length(indN) > 0
            indS = searchsortedfirst(map(x->x[4],blocklist[indN]), S_ket+insert_spin)
            val = first(indN) + indS - 1
            res[bi] = val <= length(blocklist) && blocklist[val] == S_ket+insert_spin ? val : -1
        else
            res[bi] = -1
        end
    end
    return res
end

function _build_cdag_overlap_slices(blocklist::Vector{Blockinfo})
    overlap_target = _find_cdag_overlap_blocks(blocklist)
    res = Vector{Tuple{UnitRange{Int},UnitRange{Int}}}(undef, count(overlap_target .> 0))
    j = 1
    for i in 1:length(blocklist)
        if overlap_target[i] > 0
            res[j] = (_block_slice(blocklist[i]), _block_slice(blocklist[overlap_target[i]]))
            j += 1
        end
    end
    return res
end
