# ==================================================================================================== #
#                                        GreensFunctions.jl                                            #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Measurement of Green's functions from Eigenspace.                                                  #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Only a first draft, needs refactor and testing.                                                    #
#   Do not hardcode spin up GF!!!                                                                      #
# ==================================================================================================== #


# =========================================== Overlap Matrix =========================================
"""
    overlapMatrix(es::Eigenspace, expE_list::Vector{Float64}, ket_block_index::Int)::Matrix{Float64}

Computes ``|\\langle i | c^\\dagger | j \\rangle|^2 \\frac{e^{-\\beta E_i} + e^{-\\beta E_j}}{Z}``.
TODO: not used for now, probably usefull for 2-particle GF.
"""
function overlapMatrix(es::Eigenspace, expE_list::Vector{Float64}, ket_block_index::Int)::Matrix{Float64}
    startInd_ket, blockSize_ket, Nel_ket, S_ket = es.blocklist[ket_block_index]
    indN = searchsorted(map(x->x[3],es.blocklist), Nel_ket+1)
    indS = searchsortedfirst(map(x->x[4],es.blocklist[indN]), S_ket+1)
    bra_block_index = first(indN) + indS - 1
    if bra_block_index > last(indN)  # S+1 not found!
        return Matrix{Float64}(undef, 0, 0)
    end
    startInd_bra, blockSize_bra, Nel_bra, S_bra = es.blocklist[bra_block_index]
    res = Matrix{Float64}(undef, blockSize_bra, blockSize_bra)
    #res = zeros(blockSize_bra, blockSize_bra)

    slice = startInd_bra:startInd_bra+blockSize_bra-1
    for (i,ii) in enumerate(slice)
        res[i,i] = 0.0
        for (j,jj) in enumerate((ii+1):last(slice))
            res[i, j+i] = overlap_2(es.evecs[ii],es.evecs[jj])*(expE_list[ii] + expE_list[jj])
        end
    end
    return UpperTriangular(res)
end

# ============================================ 1 Particle GF =========================================

"""
    calc_GF_1(es::Eigenspace, freq::ComplexF64, β::Float64)

Computes ``|\\langle i | c^\\dagger | j \\rangle|^2 \\frac{e^{-\\beta E_i} + e^{-\\beta E_j}}{Z (E_j - E_i + freq)}``.
TODO: not tested
"""
function calc_GF_1(es::Eigenspace, freqList::Vector{ComplexF64}, β::Float64)
    Z = calc_Z(es, β)
    expE_list = exp.(-β .* (es.evals .- es.E0))
    res = zeros(ComplexF64, length(freqList))


    for bi in 1:length(es.blocklist)
        slice = _find_overlap_block(es, bi)
        #println("block $bi:")
        for (i,ii) in enumerate(slice)
            for (j,jj) in enumerate(slice)
                val1 = overlap_2(es.evecs[ii],es.evecs[jj])
                #TODO: CDag_sign(basis.states[], i::Int)

                val2 = (expE_list[ii] + expE_list[jj])
                #println("  --> $ii, $jj: $val1 * $val2" )
                for (fi,freq) in enumerate(freqList)
                    res[fi] += val1*val2/(es.evals[jj] - es.evals[ii] + freq)
                end
            end
        end
    end
    return res ./ Z
end

"""
    _find_overlap_block(es::Eigenspace, bi::Int)::UnitRange{Int}

Find bock with [`N_el`](@ref N_el) and [`S`](@ref S) both increased by `1`.
TODO: do not hardcode spin up GF!!!
"""
function _find_overlap_block(es::Eigenspace, bi::Int)::UnitRange{Int}
    startInd_ket, blockSize_ket, Nel_ket, S_ket = es.blocklist[bi]
    indN = searchsorted(map(x->x[3],es.blocklist), Nel_ket+1)
    indS = searchsortedfirst(map(x->x[4],es.blocklist[indN]), S_ket+1)
    bra_block_index = first(indN) + indS - 1
    slice = if bra_block_index > last(indN) # S+1 not found
        0:-1
    else
        _block_slice(es.blocklist[bra_block_index])
    end
    return slice
end
