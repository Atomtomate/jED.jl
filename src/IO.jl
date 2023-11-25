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
            show(io, "text/plain", b.states[i])
            println(io, "")
        end
    end
end
# ======================= Auxilliary Function =======================

function show_matrix_block(H::AbstractMatrix, basis::Basis, iBlock::Int)
    start, size, Ni, Si = basis.blocklist[iBlock]
    slice = start:start+size-1

    println("(Block for N=$Ni, S=$Si): ")
    show(stdout, "text/plain", H[slice, slice])
    println("\n===============================")
end
