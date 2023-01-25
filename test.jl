function overlapMatrix(es::Eigenspace, blocklist, ket_block_index::Int)::Matrix{Float64}
    startInd_ket, blockSize_ket, Nel_ket, S_ket = blocklist[ket_block_index]
    indN = searchsorted(map(x->x[3],blocklist), Nel_ket+1)
    indS = searchsortedfirst(map(x->x[4],blocklist[indN]), S_ket-1)
    bra_block_index = first(indN) + indS - 1
    if bra_block_index > last(indN)  # S-1 not found!
        return Matrix{Float64}(undef, 0, 0)
    end
    #res = zeros(blockSize_bra, blockSize_bra)

    slice = jED._block_slice(blocklist[bra_block_index])
    res = Matrix{Float64}(undef, length(slice), length(slice))
    for (i,ii) in enumerate(slice)
        res[i,i] = 0.0
        for (j,jj) in enumerate((ii+1):last(slice))
            res[i, j+i] = overlap_2(es.evecs[ii],es.evecs[jj])
        end
    end
    return UpperTriangular(res)
end
