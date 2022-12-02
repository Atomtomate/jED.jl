@testset "N 2, AIM" begin
    ϵₖ = [0.1]
    Vₖ = [0.2]
    μ  = 0.6
    U  = 1.2
    β  = 13.4
    basis = jED.Basis(2)
    model = AIM(ϵₖ, Vₖ, μ, U)
    H = calc_Hamiltonian(model, basis)

    Si_check = [0,-1,-1,1,1,-2,0,0,0,0,2,-1,-1,1,1,0]
    Ni_check = [0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4]
    Si = jED.S.(basis.states)
    Ni = jED.N_el.(basis.states)

    # Basis check
    @test basis.NFlavors == 2
    @test basis.NSites == 2
    @test length(basis.blocklist) == 9
    @test all(Si .== Si_check)
    @test all(Ni .== Ni_check)

    # Model check
end
