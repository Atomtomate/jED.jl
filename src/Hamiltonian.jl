
"""
    overlap_cdagger_c(left::Fockstate, right::Fockstate, i::Int, j::Int)

Calculate ⟨left| c^†_i c_j |right⟩, i.e. `i` is the index for the creation operator
and `j` the index for the annihilation operator.
Internally, first check, that both states have exactly two or no difference.
In both cases the product of [`C_sign`](@ref C_sign) and [`CDag_sign`](@ref CDag_sign) is returned, otherwise 0.

Returns: -1/0/1
"""
function overlap_cdagger_c(left::Fockstate, right::Fockstate, i::Int, j::Int)::Int
    diff = sum(xor.(left,right))
    if diff == 2  
        CDag_sign(right,i)*C_sign(right,j)
    elseif diff == 0 && i == j
        right[j]
    else
        0
    end
end
