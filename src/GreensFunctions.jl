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
    Overlap

Contains information about overlap of Fock states under given [`Operator`](@ref Operator).

Fields
-------------
- **`op`**        : Function, operation on Fock state
- **`ov_blocks`** : Vector{Int}, index is starting block, entry target
- **`ov_list`**   : Vector{Int}, index is starting block, entry index in eigenvector (see [`Eigenspace`](@ref Eigenspace))
"""
struct Overlap
    op::Operator
    ov_blocks::Vector{Int}
    ov_list::Vector{Int}
    function Overlap(basis::Basis, op::Operator)
        ov      = _find_cdag_overlap_blocks(basis.blocklist, op)
        ov_list = _overlap_list(basis, op)
        new(op, ov, ov_list)
    end
end

# ============================================ 1 Particle GF =========================================
function overlap_EDiff(basis::Basis, es::Eigenspace{FPT}, overlap::Overlap, β::Float64, ϵ_cut::Float64; with_density::Bool=false) where FPT <: Real
    bl = basis.blocklist
    res_EDiff   = Stack{FPT}()
    res_factor  = Stack{FPT}()
    expE        = exp.(-β .* es.evals)
    dens        = 0.0
    #TODO: exclude underflow here already
    for (block_from, block_to) in enumerate(overlap.ov_blocks)
        if block_to > 0
            slice_from = _block_slice(bl[block_from])
            slice_to   = _block_slice(bl[block_to])
            block_from_start = bl[block_from][1] - 1
            block_to_start = bl[block_to][1] - 1
            for (i_from,ev_from) in enumerate(es.evecs[slice_from])
                tmp = similar(ev_from)
                for (i_to,ev_to) in enumerate(es.evecs[slice_to])
                    ov = _overlap_cdagger_ev!(tmp, ev_from, ev_to, overlap.ov_list[slice_from])^2
                    el_i = (expE[block_from_start + i_from] + expE[block_to_start + i_to])*ov
                    with_density && (dens += ov * expE[block_to_start + i_to])#expE[block_from_start + i_from])
                    if abs(el_i) > ϵ_cut
                        push!(res_factor, el_i)
                        push!(res_EDiff, es.evals[slice_from][i_from] - es.evals[slice_to][i_to])
                    end
                end
            end
        end
    end
    return collect(res_factor), collect(res_EDiff), dens
end

"""
calc_GF_1(basis::Basis, es::Eigenspace, νnGrid::AbstractVector{ComplexF64}, β::Float64; ϵ_cut::Float64=1e-16, overlap=nothing)

Computes ``|\\langle i | c^\\dagger | j \\rangle|^2 \\frac{e^{-\\beta E_i} + e^{-\\beta E_j}}{Z (E_j - E_i + freq)}``.

Arguments
-------------
- **`basis`**   : Basis, obtained with [`Basis`](@ref `Basis`).
- **`es`**      : Eigenspace, obtained with [`Eigenspace`](@ref `Eigenspace`)
- **`νnGrid`**  : AbstractVector{ComplexF64}, list of Matsubara Frequencies
- **`β`**       : Float64, inverse temperature.
- **`ϵ_cut`**   : Float64, cutoff for ``e^{-\\beta E_n}`` terms in Lehrmann representation (all contributions below this threshold are disregarded)
- **`overlap`** : Overlap, precalculated overlap between blocks of basis. Obtained with [`Overlap`](@ref `Overlap`)
"""
function calc_GF_1(basis::Basis, es::Eigenspace{FPT}, νnGrid::AbstractVector, β::Float64; ϵ_cut::Float64=1e-16, overlap=nothing, with_density::Bool=false) where FPT <: Real
    global to

    FPT !=== eltype(νnGrid) && println("Warning, element type of Eigenspace ($FPT) does not match element type of ν-Grid ($(eltype(νnGrid)))! Expect loss of precision.")

    Z = calc_Z(es, β)
    res = similar(νnGrid)
    fill!(res, 0.0)
    overlap = if overlap === nothing 
        state = 1
        op =  create_op(basis, state)
        Overlap(basis, op)
    else 
        overlap
    end
    @timeit to "pf2" pf, nf, dens = overlap_EDiff(basis, es, overlap, β, ϵ_cut, with_density=with_density)
    @timeit to "for" for νi in eachindex(νnGrid)
        νn = νnGrid[νi]
        for j in 1:length(pf)
            #TODO: reduce memory allocation overhead
            res[νi] -= sum(pf[j] / (-νn - nf[j])) / Z
        end
    end

    #print_density && println("Density = ", 2*dens/Z)
    return res, 2*dens/Z 
end


# ============================================ 2 Particle GF =========================================
function lehmann_full(basis::Basis, es::Eigenspace{FPT}, overlap::Overlap, β::Float64, ϵ_cut::Float64) where FPT <: Real
    bl = basis.blocklist
    res_ov   = Stack{FPT}()
    res_from = Stack{FPT}()
    res_to   = Stack{FPT}()
    res_ind_from = Stack{Int}()
    res_ind_to = Stack{Int}()

    expE        = exp.(-β .* es.evals)
    for (block_from, block_to) in enumerate(overlap.ov_blocks)
        if block_to > 0
            slice_from = _block_slice(bl[block_from])
            slice_to   = _block_slice(bl[block_to])
            block_from_start = bl[block_from][1] - 1
            block_to_start = bl[block_to][1] - 1
            for (i_from,ev_from) in enumerate(es.evecs[slice_from])
                tmp = similar(ev_from)
                ind_from = block_from_start + i_from
                for (i_to,ev_to) in enumerate(es.evecs[slice_to])
                    ind_to   = block_to_start + i_to
                    ov      = _overlap_cdagger_ev!(tmp, ev_from, ev_to, overlap.ov_list[slice_from])
                    el_to   = expE[ind_from] 
                    el_from = expE[ind_to]
                    push!(res_ov, ov)
                    push!(res_from, el_from)
                    push!(res_to, el_to)
                    push!(res_ind_from, ind_from)
                    push!(res_ind_to, ind_to)
                    # if abs(el_i) > ϵ_cut
                    # end
                end
            end
        end
    end
    return collect(res_ov), collect(res_from), collect(res_to), collect(res_ind_from),  collect(res_ind_to)
end
