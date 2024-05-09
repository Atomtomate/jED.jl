# ==================================================================================================== #
#                                              IO.jl                                                   #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe, Steffen Backes                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Input and output operations, including custom printing of types.                                   #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #

import Base: show
# ========================= Custom type overloads =========================
function show(io::IO, ::MIME"text/plain", f::Fockstate{Length}) where {Length}
    compact = get(io, :compact, false)
    bb = filter(!isspace, rpad(bitstring(BitVector(f)), Length, "0"))
    N = floor(Int, Length / 2)
    for i = 1:N
        du = parse(Int, bb[i])
        dd = parse(Int, bb[N+i])
        print(io, ("↑"^du) * ("O"^(1 - du)) * ("↓"^dd) * ("O"^(1 - dd)))
        (i < N) && print(io, "-")
    end
end

function show(io::IO, ::MIME"text/plain", b::Basis{Length}) where {Length}
    compact = get(io, :compact, false)
    for (bi, el) in enumerate(b.blocklist)
        block_str = " === Block $(lpad(bi,3))    [N = $(lpad(el[3],2)), S = $(lpad(el[4],3))] ==="
        println(block_str)
        for i in _block_slice(el)
            print(io, "   |          ")
            show(io, MIME"text/plain"(), b.states[i])
            println(io, "")
        end
    end
end
# ======================= Auxilliary Function =======================

function show_matrix_block(H::AbstractMatrix, b::Basis, iBlock::Int)
    start, size, Ni, Si = b.blocklist[iBlock]
    slice = start:start+size-1

    println("(Block for N=$Ni, S=$Si): ")
    show(stdout, "text/plain", H[slice, slice])
    println("\n===============================")
end

"""
    show_diag_Hamiltonian(b::Basis,  es::Eigenspace; io=stdout)

Displays eigenvalues for each block (sorted uin each block).
"""
function show_diag_Hamiltonian(b::Basis,  es::Eigenspace; io=stdout)
    for (bi, el) in enumerate(b.blocklist)
        block_str = " === Block $(lpad(bi,3))    [N = $(lpad(el[3],2)), S = $(lpad(el[4],3))] ==="
        println(io, block_str)
        println(io, "===================================================+")
        display(Diagonal(es.evals[_block_slice(el)] .+ es.E0))
        println(io, "\n====================================================")
        println(io, "")
    end
end

"""
    show_energies_states(b::Basis,  es::Eigenspace; io=stdout, eps_cut=1e-12)

Shows all eigenstates, each block is sorted by eigen energy.
Also displays the eigenvector in terms of the basis vectors for each eigenvalue.

TODO: reasonable formatting solution
"""
function show_energies_states(b::Basis,  es::Eigenspace; io=stdout, eps_cut=1e-12)
    for (bi, el) in enumerate(b.blocklist)
        block_str = " === Block $(lpad(bi,3))    [N = $(lpad(el[3],2)), S = $(lpad(el[4],3))] ==="
        println(block_str)
        bs = _block_slice(el)
        ii = sortperm(es.evals[bs])
        for i in bs[ii]
            print(io, "   | [E=$(lpad(round(es.evals[i] .+ es.E0; digits=4),10))]        ")
            println(io, "")
            print(io, "       |> ")
            start_of_print = true
            for (j,ev_j) in enumerate(es.evecs[i])
                if abs(ev_j) > eps_cut
                    if start_of_print == false
                        print(io, " + ")
                    end
                    start_of_print = false
                    print(io,"$(lpad(round(ev_j; digits=2),5)) x [")
                    show(io, MIME"text/plain"(), b.states[bs[ii][1]+j-1])
                    print(io,"]")
                end
            end
            println(io, "")
        end
    end
end