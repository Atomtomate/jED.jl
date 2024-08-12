# ==================================================================================================== #
#                                              IO.jl                                                   #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Steffen Backes, Julian Stobbe                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Input and output operations, including custom printing of types.                                   #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #

"""
    calc_Z(eigenspace, β)

Calculates the partition function from [`eigenspace`](@ref Eigenspace) and the inverse temperature `β`.
"""
function calc_Z(eigenspace::Eigenspace, β::Float64)
    return sum(exp.(-β .* (eigenspace.evals)))
end

"""
   calc_E(eigenspace, β)

Calculates the total energy from [`eigenspace`](@ref Eigenspace) and the inverse temperature `β`.
"""
function calc_E(eigenspace::Eigenspace, β::Float64)
    return sum(eigenspace.evals .* exp.(-β .* (eigenspace.evals))) / calc_Z(eigenspace, β)
end
#
# """
#     getN(eigenspace, basis, β)
#
# Calculates the expectation value of the occupation from  [`eigenspace`](@ref Eigenspace), [`Basis`](@ref Basis) and the inverse temperature `β`.
# #TODO: NOT IMPLEMENTED YET
# """
# function getN(eigenspace::Eigenspace, basis::Basis, beta::Float64)
#     Nmax = fockstates.norb*2
#     Z = getZ(eigenspace, beta)
#     n_ms = zeros(Float64,fockstates.norb*2)  # Filling per orbital and spin
#     for n=0:Nmax
#         for s=1:noSpinConfig(n,Nmax)
#             dim = length(eigenspace.evals[n+1][s])
#             for i=1:dim
#                 for j=1:dim
#                     n_ms += abs(eigenspace.evecs[n+1][s][i][j])^2 .* fockstates.states[n+1][s][j] * exp(-beta*(eigenspace.evals[n+1][s][i]-eigenspace.E0))
#                 end # j
#             end # i
#         end # s
#     end # n
#     return n_ms/Z
# end
#
# """
#  getNN(eigenspace)
#
# Calculate expectation value of all double occupation combinations
# """

"""
    calc_D(es::Eigenspace, β::Float64, basis::Basis{Length}, index::Int)::Float64 where Length


Calculates double occupancy of site `index`.
"""
 function calc_D(es::Eigenspace, β::Float64, basis::Basis{Length}, index::Int)::Float64 where Length
    Z = calc_Z(es, β)
    D = 0.0
    for bl_i in 1:length(basis.blocklist)
        sl = _block_slice(basis.blocklist[bl_i])
        for ii in sl
            #TODO: onlz caclcuate this once
            for (j,jj) in enumerate(sl)
                if basis.states[jj][index] & basis.states[jj][index + Int(Length / 2)]
                    D += es.evecs[ii][j] ^ 2 *  exp.(-β * es.evals[ii])
                end
            end
        end
    end
    return D/Z
 end



 """
    calc_Nup(es::Eigenspace, β::Float64, basis::Basis{Length}, index::Int)::Float64 where Length

Calculates density of site `index` for up electrons.
"""
 function calc_Nup(es::Eigenspace, β::Float64, basis::Basis{Length}, index::Int)::Float64 where Length
    Z::Float64 = calc_Z(es, β)
    N::Float64 = 0.0

    for bl_i in 1:length(basis.blocklist)
        sl = _block_slice(basis.blocklist[bl_i])
        for ii in sl
            for (j,jj) in enumerate(sl)
                if basis.states[jj][index]
                    N += es.evecs[ii][j] ^ 2 *  exp.(-β * es.evals[ii])
                end
            end
        end
    end
    return N/Z
 end



 """
 calc_Ndown(es::Eigenspace, β::Float64, basis::Basis{Length}, index::Int)::Float64 where Length

Calculates density of site `index` for down electrons.
"""
function calc_Ndo(es::Eigenspace, β::Float64, basis::Basis{Length}, index::Int)::Float64 where Length
 Z::Float64 = calc_Z(es, β)
 N::Float64 = 0.0

 for bl_i in 1:length(basis.blocklist)
     sl = _block_slice(basis.blocklist[bl_i])
     for ii in sl
         for (j,jj) in enumerate(sl)
             if basis.states[jj][index + Int(Length / 2)]
                 N += es.evecs[ii][j] ^ 2 *  exp.(-β * es.evals[ii])
             end
         end
     end
 end
 return N/Z
end

function calc_EKin_DMFT(iνₙ, ϵₖ, Vₖ, GImp, n, U, β, μ)
    E_kin = 0.0
    vk = sum(Vₖ .^ 2)
    E_kin_tail = vk
    for n in axes(GImp,1)
        Δ_n = sum((Vₖ .^ 2) ./ (iνₙ[n] .- ϵₖ))
        E_kin += 2*real(GImp[n] * Δ_n - E_kin_tail/(iνₙ[n]^2))
    end
    E_kin = E_kin .* (2/β) - (β/2) .* E_kin_tail
    return E_kin
end

function calc_EPot_DMFT(iνₙ, ϵₖ, Vₖ, GImp, n, U, β, μ)
    E_pot = 0.0
    Σ_hartree = n * U/2
    E_pot_tail = (U^2)/2 * n * (1-n/2) - Σ_hartree*(Σ_hartree-μ)

    for n in axes(GImp,1)
        Δ_n = sum((Vₖ .^ 2) ./ (iνₙ[n] .- ϵₖ))
        Σ_n = iνₙ[n] .- Δ_n .- 1.0 ./ GImp[n] .+ μ
        E_pot += 2*real(GImp[n] * Σ_n - E_pot_tail/(iνₙ[n]^2))
    end
    E_pot = E_pot .* (1/β) .+ 0.5*Σ_hartree .- (β/4) .* E_pot_tail
    return E_pot
end
