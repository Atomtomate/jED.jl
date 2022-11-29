# ==================================================================================================== #
#                                            States.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Type definition and basic operations on Fock states.                                               #
# -------------------------------------------- TODO -------------------------------------------------- #
#   _generate_blocks refactor                                                                          #
# ==================================================================================================== #


const Fockstate = SVector{Length, Bool} where Length

"""
    Basis

A state is represented as a BitVetor of length 64. The first 32 bit (ordering follows Julia Vector indexing) correspond to spin up for each site (index 1 corresponds to the first site, 2 to the second, etc.) and bits 33 to 64 to spin down, for site 1 to 32.
Each bit is set to 1 if it is occupied by an electron of the given spin.
In principle, the number of electron flavors and orbitals is free, however, for now it is fixed to 2 and 1.

A list of states should be generated using `States(NSites)` with NSites being the number of sites in the model.

Fields
-------------
- **`NFlavors`**  : Integer, number of flavors (TODOhardcoded to `2` for now!)
- **`NSites`**    : Integer, number of sites
- **`states`**    : Vector{State}, list of Fock states. The are sorted according to good quantum numbers. `blocklist` contains start and size of blocks.
- **`blocklist`** : Vector{Tuple{Int,Int}}, Vector of tuples. Each entry encodes a block with the first element being the start and the second the column size of the block 
"""
struct Basis{Length}
    NFlavors::Int
    NSites::Int
    states::Vector{Fockstate{Length}}
    blocklist::Vector{Tuple{Int,Int}}
end

function Basis(NSites::Int; NFlavors::Int = 2)
    Length = NFlavors * NSites
    states = Vector{Fockstate{Length}}(undef, 4^NSites)
    NInt = 2^NSites - 1
    ii = 1
    for i_up in 0:NInt
        for i_down in 0:NInt
            # states[ii] = BitVector(cat(digits(i_up,base=2,pad=32), digits(i_down,base=2,pad=32),dims=1))
            states[ii] = Fockstate{Length}(cat(digits(i_up,  base=2, pad=NSites),
                                               digits(i_down,base=2, pad=NSites),
                                               dims=1))
            ii += 1
        end
    end

    blocklist = _generate_blocks!(states)

    Basis{Length}(NFlavors, NSites, states, blocklist)
end

"""
    N_el(s::AbstractVector)::Int

Total electron number of state.
"""
N_el(s::Fockstate)::Int = sum(s)

"""
    S(s::AbstractVector)::Int

Total spin of state.
#TODO: only implemented for flavor=2
"""
S(s::Fockstate{NSites}) where NSites = sum(s[1:Int(NSites/2)]) - sum(s[Int(NSites/2)+1:end])

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
C_sign(state::Fockstate, i::Int) = (1-2*(sum(state[1:i-1])%2))*state[i]

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
CDag_sign(state::Fockstate, i::Int) = (1-2*(sum(state[1:i-1])%2))*(!state[i])


# ============================================= Internals ============================================

"""
    _generate_blocks!(states::Vector{Fockstate})

Sort state list and generate list of blocks.
"""
function _generate_blocks!(states::Vector{Fockstate{Length}}) where Length
    sort_f(x::Fockstate) = N_el(x)^3 + S(x)
    sort!(states, by = sort_f)
    blocks = Vector{Tuple{Int, Int}}(undef, 0)

    last_N::Int = N_el(states[1])
    last_S::Int = S(states[1])
    current_block_start = 1
    current_block_size = 1

    for i in 2:length(states)
        Ni = N_el(states[i])
        Si = S(states[i])
        if Ni != last_N || Si != last_S
            push!(blocks, (current_block_start, current_block_size))
            last_N = Ni
            last_S = Si
            current_block_size = 0
            current_block_start = i
        end
        current_block_size += 1
    end
    push!(blocks, (current_block_start, current_block_size))
    return blocks
end
