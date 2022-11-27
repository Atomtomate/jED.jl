const Fockstate = SVector

"""
    States

A state is represented as a BitVetor of length 64. The first 32 bit (ordering follows Julia Vector indexing) correspond to spin up for each site (index 1 corresponds to the first site, 2 to the second, etc.) and bits 33 to 64 to spin down, for site 1 to 32.
Each bit is set to 1 if it is occupied by an electron of the given spin.
In principle, the number of electron flavors and orbitals is free, however, for now it is fixed to 2 and 1.

A list of states should be generated using `States(NSites)` with NSites being the number of sites in the model.
"""
struct States{Length}
    NFlavors::Int
    NSites::Int
    states::Vector{SVector{Length, Bool}}
end

function States(NSites::Int; NFlavors::Int = 2)
    Length = NFlavors * NSites
    states = Vector{SVector{Length}}(undef, 4^NSites)
    NInt = 2^NSites - 1
    ii = 1
    for i_up in 0:NInt
        for i_down in 0:NInt
            # states[ii] = BitVector(cat(digits(i_up,base=2,pad=32), digits(i_down,base=2,pad=32),dims=1))
            states[ii] = SVector{Length}(cat(digits(i_up,  base=2, pad=NSites),
                                             digits(i_down,base=2, pad=NSites),
                                             dims=1))
            ii += 1
        end
    end
    States{Length}(NFlavors, NSites, states)
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
S(s::Fockstate, NSites::Int)::Int = sum(s[1:NSites]) - sum(s[NSites+1:end])

"""
    C_sign(state,i)

Sign when creating an electron at position `i` in `state`.
Returns: -1/1/0 for Uneven/Even permutations and 0 when the state is not occupied.
```
julia> C_sign(SVector{6}(Bool[1,0,1,1,0,0]),3)
-1
julia> C_sign(SVector{6}(Bool[1,0,1,1,0,0]),2)
0
julia> C_sign(SVector{6}(Bool[1,0,1,1,0,0]),4)
```
"""
C_sign(state::Fockstate, i::Int) = (1-2*(sum(state[1:i-1])%2))*state[i]

"""
    CDag_sign(state,i)

Sign when annihilating an electron at position `i` in `state`.
Returns: -1/1/0 for Uneven/Even permutations and 0 when the state is not occupied.
```
julia> CDag_sign(SVector{6}(Bool[1,0,1,1,0,0]),5)
-1
julia> CDag_sign(SVector{6}(Bool[1,0,1,1,0,0]),3)
0
julia> CDag_sign(SVector{7}(Bool[1,0,1,0,1,0,0]),4)
1
```
"""
CDag_sign(state::Fockstate, i::Int) = (1-2*(sum(state[1:i-1])%2))*(1-state[i])
