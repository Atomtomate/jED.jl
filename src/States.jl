# ==================================================================================================== #
#                                            States.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Type definition and basic operations on Fock states.                                               #
# -------------------------------------------- TODO -------------------------------------------------- #
#   store C,CDag signs in Basis
#   _generate_blocks refactor                                                                          #
# ==================================================================================================== #


"""
    Fockstate

Internal representation of a Fockstate.

Fockstates of a basis with `N` electrons are represented by a vector of `2N` Booleans.
The first half is 1, if the corresponding state has a spin up electron at that position,
the second half corresponds to spin down electrons.
"""
const Fockstate = SVector{Length,Bool} where {Length}
const Blockinfo = NTuple{4,Int}

# =============================================== Basis ==============================================
"""
    Basis

A state is represented as a BitVetor of length 64. The first 32 bit (ordering follows Julia Vector indexing) correspond to spin up for each site (index 1 corresponds to the first site, 2 to the second, etc.) and bits 33 to 64 to spin down, for site 1 to 32.
Each bit is set to 1 if it is occupied by an electron of the given spin.
In principle, the number of electron flavors and orbitals is free, however, for now it is fixed to 2 and 1.

A list of states should be generated using `States(NSites)` with NSites being the number of sites in the model.

Fields
-------------
- **`NFlavors`**  : Integer, number of flavors (TODO: hardcoded to `2` for now. later: orbitals*2?!)
- **`NSites`**    : Integer, number of sites
- **`states`**    : Vector{State}, list of Fock states. The are sorted according to good quantum numbers. `blocklist` contains start and size of blocks.
- **`blocklist`** : Vector{NTuple{4,Int}}, Vector of 4-tuples. Each entry encodes a block in the following form: 1. element is the start index of the block, 2. element is the column size of the block, 3. element is the electron number (see [`N_el`](@ref N_el)), 4. element is the spin (see [`S`](@ref S)) 
- **`cdag_ov`**   : Vector{Tuple{UnitRange{Int},UnitRange{Int}}}, Slices of states, that contribute to an overlap with one creation operator. Needed for performance reasons in computation of Lehmann representation (see also (@ref _find_cdag_overlap_blocks)[`_find_cdag_overlap_blocks`])).
"""
struct Basis{Length}
    NFlavors::Int
    NSites::Int
    states::Vector{Fockstate{Length}}
    blocklist::Vector{Blockinfo}
end

"""
    Basis(NSites::Int; NFlavors::Int=2; N_filter=nothing, S_filter=nothing)

Contructs a Fock basis for `NSites` with `NFlavors`.
Provide `N_filter` or `S_filter` if you want only thos quantum number is the basis.
"""
function Basis(NSites::Int; NFlavors::Int=2, N_filter=Int[], S_filter=Int[])
    Length = NFlavors * NSites
    states = Vector{Fockstate{Length}}(undef, 4^NSites)
    NInt = 2^NSites - 1
    ii = 1
    for i_up = 0:NInt
        for i_down = 0:NInt
            # states[ii] = BitVector(cat(digits(i_up,base=2,pad=32), digits(i_down,base=2,pad=32),dims=1))
            states[ii] = Fockstate{Length}(
                cat(
                    digits(i_up, base=2, pad=NSites),
                    digits(i_down, base=2, pad=NSites),
                    dims=1,
                ),
            )
            ii += 1
        end
    end
    blocklist = _generate_blocks!(states, N_filter=N_filter, S_filter=S_filter)
    Basis{Length}(NFlavors, NSites, states, blocklist)
end

"""
    N_el(s::AbstractVector)::Int

Total electron number of state.
"""
N_el(s::Fockstate)::Int = sum(s)

"""
    N_up(s::Fockstate{NSites})

Number of up electrons, ``N_\\uparrow`` in state `s`.
"""
N_up(s::Fockstate{NSites}) where {NSites} = sum(s[1:Int(NSites / 2)])

"""
    N_do(s::Fockstate{NSites})

Number of down electrons, ``N_\\downarrow`` in state `s`.
"""
N_do(s::Fockstate{NSites}) where {NSites} = sum(s[Int(NSites / 2)+1:end])

"""
    S(s::Fockstate{NSites}) where {NSites}

Total spin of state.
#TODO: only implemented for flavor=2
"""
S(s::Fockstate{NSites}) where {NSites} = N_up(s) - N_do(s)

"""
    C_sign(state,i)

Sign when creating an electron at position `i` in `state`.
Returns: -1/1/0 for Uneven/Even permutations and 0 when the state is not occupied.
```
julia> C_sign(Fockstate{6}(Bool[1,0,1,1,0,0]),3)
-1
julia> C_sign(Fockstate{6}(Bool[1,0,1,1,0,0]),2)
0
julia> C_sign(Fockstate{6}(Bool[1,0,1,1,0,0]),4)
```
"""
C_sign(state::Fockstate, i::Int) = (1 - 2 * (sum(state[1:i-1]) % 2)) * state[i]

"""
    CDag_sign(state,i)

Sign when annihilating an electron at position `i` in `state`.
Returns: -1/1/0 for Uneven/Even permutations and 0 when the state is not occupied.
```
julia> CDag_sign(Fockstate{6}(Bool[1,0,1,1,0,0]),5)
-1
julia> CDag_sign(Fockstate{6}(Bool[1,0,1,1,0,0]),3)
0
julia> CDag_sign(Fockstate{7}(Bool[1,0,1,0,1,0,0]),4)
1
```
"""
CDag_sign(state::Fockstate, i::Int) = (1 - 2 * (sum(state[1:i-1]) % 2)) * (!state[i])


# ============================================= Internals ============================================

"""
    _generate_blocks!(states::Vector{Fockstate})

Sort state list and generate list of blocks.
"""
function _generate_blocks!(states::Vector{Fockstate{Length}}; N_filter = Int[], S_filter = Int[]) where {Length}
    sort_f(x::Fockstate) = N_el(x)*2*Length + S(x)
    filter!(x->    (isempty(N_filter) || (N_el(x) in N_filter)) 
                && (isempty(S_filter) || (S(x) in S_filter)), states)
    isempty(states) && throw(ArgumentError("Filter condition for quantum numbers produced empty basis!"))
    sort!(states, by=sort_f)
    blocks = Vector{Blockinfo}(undef, 0)

    last_N::Int = N_el(states[1])
    last_S::Int = S(states[1])
    current_block_start = 1
    current_block_size = 1

    for i = 2:length(states)
        Ni = N_el(states[i])
        Si = S(states[i])
        if Ni != last_N || Si != last_S
            push!(blocks, (current_block_start, current_block_size, last_N, last_S))
            last_N = Ni
            last_S = Si
            current_block_size = 0
            current_block_start = i
        end
        current_block_size += 1
    end
    Ni = N_el(states[end])
    Si = S(states[end])
    push!(blocks, (current_block_start, current_block_size, Ni, Si))
    return blocks
end

"""
    _block_slice(bi::Blockinfo)

Slice of continuous vector for given block `bi`.
"""
_block_slice(bi::Blockinfo) = bi[1]:bi[1]+bi[2]-1