# ==================================================================================================== #
#                                          Eigenspace.jl                                               #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Types and constructors for an Eigenspace, given a model Hamiltonian and states.                    #
# -------------------------------------------- TODO -------------------------------------------------- #
#   _H_nn is not complete (J term missing)                                                             #
# ==================================================================================================== #

const Hamiltonian     = Hermitian{ComplexF64, Matrix{ComplexF64}}
const HamiltonianReal = Symmetric{Float64,    Matrix{Float64}}


# ============================================= Eigenspace ===========================================
"""
    Eigenspace

Containes Eigenvalues and Eigenvectors grouped into blocks with are each ordered by magnitude of Eigenvalues.
The blocks are part of the [`Basis`](@ref Basis).

Fields
-------------
- **`evals`**     : Eigenvalues
- **`evecs`**     : Eigenvectors
- **`blocklist`** : List of start indices and lengths of blocks.
- **`E0`**        : smallest Eigenvalue
"""
struct Eigenspace{T}
    evals::Vector{Float64}
    evecs::Vector{Vector{T}}
    E0::Float64
end

"""

Constructs [`Eigenspace`](@ref Eigenspace) for [`Model`](@ref Model) over given [`Basis`](@ref Basis) by diagonalizing the Hamiltonian (see also [`calc_Hamiltonian`](@ref calc_Hamiltonian)) for each block.
"""
function Eigenspace(model::Model, basis::Basis)

    EVecType = typeof(model).parameters[2]
    evals = Vector{Float64}(undef, length(basis.states))
    evecs = Vector{Vector{EVecType}}(undef, length(basis.states))

    print("Generating Eigenspace:   0.0% done.")
    for el in basis.blocklist
        slice = _block_slice(el)
        Hi = calc_Hamiltonian(model, basis.states[slice])
        tmp = eigen(Hi, sortby=nothing)
        evals[slice] .= tmp.values
        for i in 1:length(tmp.values)
            evecs[first(slice)+i-1] = tmp.vectors[:,i]
        end
        done = lpad(round(100*(el[1]+el[2])/length(basis.states), digits=1), 5, " ")
        print("\rGenerating Eigenspace: $(done)% done.")
    end
    println("\rEigenspace generated!                  ")
    E0 = minimum(evals)
    
    return Eigenspace{EVecType}(evals, evecs, E0)
end

# ============================================ Hamiltonian ===========================================
"""
    calc_Hamiltonian(model::Model, basis::Basis)

Calculates the Hamiltonian for a given 
  - `model`, see for example [`AIM`](@ref AIM))) in a 
  - `basis`, see [`Basis`](@ref Basis)  
"""
function calc_Hamiltonian(model::Model, states::Vector{Fockstate{NSites}}) where NSites
    Hsize = length(states)
    T = eltype(model.tMatrix)
    H_int = Matrix{T}(undef, Hsize, Hsize)
    for i in 1:Hsize
        H_int[i,i] = _H_nn(states[i], states[i], model.UMatrix) + _H_CDagC(states[i], states[i], model.tMatrix)
        # We are generating a Hermitian/Symmetric matrix and only need to store the upper triangular part
        for j in i+1:Hsize
            val = _H_CDagC(states[i], states[j], model.tMatrix)
            H_int[i,j] = val
        end
    end
    return T === ComplexF64 ? Hermitian(H_int, :U) : Symmetric(H_int, :U)
end

calc_Hamiltonian(model::Model, basis::Basis) = calc_Hamiltonian(model, basis.states) 



# ======================================== Auxilliary Functions ======================================
"""
    _H_CDag_C(istate, jstate, tMatrix)

Returns the hopping contribution for states ``\\sum_{i,j} \\langle i | T | j \\rangle``, with T being the hopping matrix `tmatrix` and
the states i and j given by `istate` and `jstate`.
"""
function _H_CDagC(bra::Fockstate, ket::Fockstate, tMatrix::SMatrix)
    T = eltype(tMatrix)
    res::T   = zero(T)
    NFlavors = 2
    NSites = size(tMatrix,1)

    for i=1:NSites
        for j=1:NSites
            tval = tMatrix[i,j]
            if tval != 0
                for f in 1:NFlavors
                    annInd = NSites * (f-1) + i
                    createInd = NSites * (f-1) + j
                    res += tval*overlap_cdagger_c(bra, createInd, ket, annInd)
                end
            end
        end
    end
    return res
end

"""
    _H_nn(istate, jstate, tMatrix)

Returns the hopping contribution for states ``\\sum_{i,j} \\langle i | T | j \\rangle``, with T being the hopping matrix `tmatrix` and
the states i and j given by `istate` and `jstate`.
"""
function _H_nn(bra::Fockstate, ket::Fockstate, UMatrix::SMatrix)
    T = eltype(UMatrix)
    res::T   = zero(T)
    NFlavors = 2
    NSites   = size(UMatrix,1)

    for i=1:NSites
        uval = UMatrix[i,i]
        i1 = i
        i2 = NSites + i
        n1 = overlap_ni_nj(bra, ket, i1, i2)
        res += uval*n1
    end
    return res
end
