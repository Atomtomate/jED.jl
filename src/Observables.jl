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
    return sum(exp.(-β .* (eigenspace.evals .- eigenspace.E0)))
end

"""
   calc_E(eigenspace, β)

Calculates the total energy from [`eigenspace`](@ref Eigenspace) and the inverse temperature `β`.
"""
function calc_E(eigenspace::Eigenspace, β::Float64)
    return sum(eigenspace.evals .* exp.(-β .* (eigenspace.evals .- eigenspace.E0))) / calc_Z(eigenspace, β)
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
#     getNN(eigenspace)
#
# Calculate expectation value of all double occupation combinations
# """
# function getNN(eigenspace::Eigenspace, beta::Float64, fockstates::Fockstates)::Array{Float64,3}
#     Nmax = fockstates.norb*2
#     Z = getZ(eigenspace, beta)
#     nn = zeros(Float64,3,fockstates.norb,fockstates.norb)  # Nup*Ndn, Nup*Nup, Ndn*Ndn as orbital matrix
#     for n=0:Nmax
#         for s=1:noSpinConfig(n,Nmax)
#             dim = length(eigenspace.evals[n+1][s])
#             for i=1:dim
#                 for j=1:dim
#                     weight = abs(eigenspace.evecs[n+1][s][i][j])^2 * exp(-beta*(eigenspace.evals[n+1][s][i]-eigenspace.E0)) # entry in Eigenvector * Boltzmann factor
#                     for m1=1:fockstates.norb
#                         for m2=1:fockstates.norb
#                             nn[1,m1,m2] += weight * fockstates.states[n+1][s][j][2*m1-1] * fockstates.states[n+1][s][j][2*m2-0] 
#                             nn[2,m1,m2] += weight * fockstates.states[n+1][s][j][2*m1-1] * fockstates.states[n+1][s][j][2*m2-1] 
#                             nn[3,m1,m2] += weight * fockstates.states[n+1][s][j][2*m1-0] * fockstates.states[n+1][s][j][2*m2-0] 
#                         end # m2
#                     end # m1
#                 end # j
#             end # i
#         end # s
#     end # n
#     return nn/Z
# end
