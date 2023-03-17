"""
    overlap_cdagger(basis::Basis, es::Eigenspace, block_overlaps::Vector{Int}; state = 1) 

Computes the overlap of eigenvectors from the Hamiltonian, given a creation operator in Fock basis: ``\\langle E_i | c^\\dagger | E_j \\rangle``.
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

"""
    prefactor(basis::Basis, es::Eigenspace, block_overlaps::Vector{Int}, β::Float64)


"""
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


"""
    E_diff(basis::Basis, es::Eigenspace, block_overlaps::Vector{Int})

"""
function E_diff(basis::Basis, es::Eigenspace, block_overlaps::Vector{Int})
    bl = basis.blocklist
    #TODO: overlaps can be 1D list, when list of evals is computed as well
    res = Vector{Matrix{ComplexF64}}(undef, length(bl))
    for (block_from, block_to) in enumerate(block_overlaps)
        if block_to > 0
            slice_from = _block_slice(bl[block_from])
            slice_to = _block_slice(bl[block_to])
            res[block_from] = (es.evals[slice_from] .- es.evals[slice_to]')
        else
            res[block_from] = Matrix{Float64}(undef, 0,0)
        end
    end
    return res
end

function νfactor(basis::Basis, es::Eigenspace, block_overlaps::Vector{Int}, ν::ComplexF64)
    bl = basis.blocklist
    #TODO: overlaps can be 1D list, when list of evals is computed as well
    prefactor = Vector{Matrix{ComplexF64}}(undef, length(bl))
    for (block_from, block_to) in enumerate(block_overlaps)
        if block_to > 0
            slice_from = _block_slice(bl[block_from])
            slice_to = _block_slice(bl[block_to])
            prefactor[block_from] = ν .- (es.evals[slice_from] .- es.evals[slice_to]')
        else
            prefactor[block_from] = Matrix{Float64}(undef, 0,0)
        end
    end
    return prefactor
end
function calc_GF_1_full(basis::Basis, es::Eigenspace, νnGrid::AbstractVector{ComplexF64}, β::Float64)
    global to
    Z = calc_Z(es, β)
    res = similar(νnGrid)
    fill!(res, 0.0)
    state = 1
    op =  create_op(basis, state)
    ov = _find_cdag_overlap_blocks(basis.blocklist, op)
    @timeit to "lm" lm = overlap_cdagger(basis, es, ov)
    @timeit to "pf" pf = prefactor(basis, es, ov, β)
    for νi in eachindex(νnGrid)
        νn = νnGrid[νi]
        @timeit to "nf" nf = νfactor(basis, es, ov, νn)
        for j in 1:length(lm)
            res[νi] += sum(lm[j] .* pf[j] ./ transpose(nf[j]))
        end
    end
    return -conj.(res) ./ Z
end

