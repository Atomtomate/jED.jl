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
    overlapMatrix(es::Eigenspace, ket_block_index::Int)::Matrix{Float64}

Computes ``|\\langle i | c^\\dagger | j \\rangle|^2.
TODO: not used for now, probably usefull for 2-particle GF.
"""
function overlapMatrix(es::Eigenspace, ket_block_index::Int)::Matrix{Float64}
    startInd_ket, blockSize_ket, Nel_ket, S_ket = es.blocklist[ket_block_index]
    indN = searchsorted(map(x->x[3],es.blocklist), Nel_ket+1)
    indS = searchsortedfirst(map(x->x[4],es.blocklist[indN]), S_ket-1)
    bra_block_index = first(indN) + indS - 1
    if bra_block_index > last(indN)  # S-1 not found!
        return Matrix{Float64}(undef, 0, 0)
    end
    res = Matrix{Float64}(undef, blockSize_bra, blockSize_bra)
    #res = zeros(blockSize_bra, blockSize_bra)

    slice = _block_slice(es.blocklist[bra_block_index])
    for (i,ii) in enumerate(slice)
        res[i,i] = 0.0
        for (j,jj) in enumerate((ii+1):last(slice))
            res[i, j+i] = overlap_2(es.evecs[ii],es.evecs[jj])
        end
    end
    return UpperTriangular(res)
end

# ============================================ 1 Particle GF =========================================
# TODO: refactor: loop over block overlaps is repeated twice
"""

"""
function overlap_cdagger(basis::Basis, es::Eigenspace, block_overlaps::Vector{Int}; state = 1) 
    # state = 1 <=> spin down impurity
    op = create_op(basis, state)
    bl = basis.blocklist
    #TODO: overlaps can be 1D list, when list of evals is computed as well
    overlaps = Vector{Matrix{Float64}}(undef, length(bl))
    for (block_from, block_to) in enumerate(block_overlaps)
        if block_to > 0
            len_from = bl[block_from][2]
            len_to = bl[block_to][2]
            slice_from = _block_slice(bl[block_from])
            slice_to = _block_slice(bl[block_to])
            ov = _overlap_list(basis, block_from, block_to, op)
            overlaps[block_from] = Matrix{Float64}(undef, len_to,len_from)
            for (i_from,ev_from) in enumerate(es.evecs[slice_from])
                tmp = similar(ev_from)
                for (i_to,ev_to) in enumerate(es.evecs[slice_to])
                    overlaps[block_from][i_to,i_from] = _overlap_cdagger_ev!(tmp, ev_from, ev_to, ov)^2
                end
            end
        else
            overlaps[block_from] = Matrix{Float64}(undef, 0,0)
        end
    end
    return overlaps
end

function prefactor(basis::Basis, es::Eigenspace, block_overlaps::Vector{Int}, β::Float64)
    bl = basis.blocklist
    #TODO: overlaps can be 1D list, when list of evals is computed as well
    prefactor = Vector{Matrix{Float64}}(undef, length(bl))
    expE = exp.(-β .* es.evals)
    for (block_from, block_to) in enumerate(block_overlaps)
        if block_to > 0
            slice_from = _block_slice(bl[block_from])
            slice_to = _block_slice(bl[block_to])
            prefactor[block_from] = expE[slice_from]' .+ expE[slice_to] 
        else
            prefactor[block_from] = Matrix{Float64}(undef, 0,0)
        end
    end
    return prefactor
end


function νfactor(basis::Basis, es::Eigenspace, block_overlaps::Vector{Int})
    bl = basis.blocklist
    #TODO: overlaps can be 1D list, when list of evals is computed as well
    prefactor = Vector{Matrix{ComplexF64}}(undef, length(bl))
    for (block_from, block_to) in enumerate(block_overlaps)
        if block_to > 0
            slice_from = _block_slice(bl[block_from])
            slice_to = _block_slice(bl[block_to])
            prefactor[block_from] = (es.evals[slice_from] .- es.evals[slice_to]')
        else
            prefactor[block_from] = Matrix{Float64}(undef, 0,0)
        end
    end
    return prefactor
end
"""
    calc_GF_1(es::Eigenspace, freq::ComplexF64, β::Float64)

Computes ``|\\langle i | c^\\dagger | j \\rangle|^2 \\frac{e^{-\\beta E_i} + e^{-\\beta E_j}}{Z (E_j - E_i + freq)}``.
TODO: not tested
"""
function calc_GF_1(basis::Basis, es::Eigenspace, νnGrid::AbstractVector{ComplexF64}, β::Float64; prefac_cut::Float64=0.0)
    global to
    Z = calc_Z(es, β)
    res = similar(νnGrid)
    fill!(res, 0.0)
    state = 1
    op =  create_op(basis, state)
    ov = _find_cdag_overlap_blocks(basis.blocklist, op)
    lm = overlap_cdagger(basis, es, ov)
    pf = prefactor(basis, es, ov, β)
    nf = νfactor(basis, es, ov)
    prefactors    = [lm[j] .* pf[j] for j in 1:length(lm)]
    valid_indices = findall(x-> length(x) > 0 && maximum(abs.(x)) >= prefac_cut, prefactors)
    for νi in eachindex(νnGrid)
        νn = νnGrid[νi]
        for j in valid_indices
            #TODO: reduce memory allocation overhead
            res[νi] += sum(prefactors[j] ./ transpose(νn .- nf[j]))
        end
    end
    return -conj.(res) ./ Z
end
