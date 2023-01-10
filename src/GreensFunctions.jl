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


function _overlap(es::Eigenspace, basis::Basis, i::Int, freq, β)

    overlap_indices = _find_cdag_overlap_index_naive(basis, 1)
    res = zeros(ComplexF64,length(ket_slice), length(bra_slice))
    for bra_i in es.evecs

    end
    for (i,ket_i) in enumerate(es.evecs[ket_slice])
        for (j,bra_i) in enumerate(es.evecs[bra_slice])
            tmp = 0.0
            len = min(length(ket_i),length(bra_i))
            for ki in 1:len
                for bi in 1:len
                    #println(i, " ", j, ": ", ki, "*", bi)
                    tmp += (ket_i[ki]*bra_i[bi])^2
                end
            end
            res[i,j] += tmp
            #res[i,j] += (exp(-β * es.evals[ket_slice[i]]) + exp(- β * es.evals[bra_slice[j]])) * tmp^2 / ( es.evals[ket_slice[i]] - es.evals[bra_slice[j]] + freq)
            #res += (exp(-β * es.evals[bra_i]) + exp(-β * es.evals[ket_i])) * tmp^2 / (es.evals[bra_i] - es.evals[ket_i] + freq)
        end
    end
        return res
end

"""
    calc_GF_1(es::Eigenspace, freq::ComplexF64, β::Float64)

Computes ``|\\langle i | c^\\dagger | j \\rangle|^2 \\frac{e^{-\\beta E_i} + e^{-\\beta E_j}}{Z (E_j - E_i + freq)}``.
TODO: not tested
"""
function calc_GF_1(es::Eigenspace, basis::Basis, freqList::Vector{ComplexF64}, β::Float64)
    Z = calc_Z(es, β)
    expE_list = exp.(-β .* (es.evals .- es.E0))
    res = zeros(ComplexF64, length(freqList))

    for i in 1:length(es.cdag_ov)
        for (fi, freq) in enumerate(freqList)
            res[fi] += sum(_overlap(es, i, freq, β))
        end
    end
    return res ./ Z
end
