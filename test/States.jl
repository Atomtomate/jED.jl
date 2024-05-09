@testset "auxilliary functions" begin
    s = jED.SVector{8}(Bool[1,0,1,0,1,0,0,0])
    @test jED.N_el(s) == 3
    @test jED.N_up(s) == 2
    @test jED.N_do(s) == 1
    @test jED.S(s) == 1
    @test jED.C_sign(s,3) == -1
    @test jED.C_sign(s,2) == 0
    @test jED.C_sign(s,5) == 1
    @test jED.CDag_sign(s,2) == -1
    @test jED.CDag_sign(s,3) == 0
    @test jED.CDag_sign(s,4) == 1
end

@testset "Basis" begin
    for NSites in 2:5
        s = jED.Basis(NSites)
        @test s.NFlavors == 2
        @test s.NSites == NSites
        @test length(s.states) == 4^NSites
        Nel_arr_tmp = sort(jED.N_el.(s.states))
        Nel_arr = [count(x -> i == x, Nel_arr_tmp) for i in unique(Nel_arr_tmp)]
        @test all(Nel_arr .== [binomial(2*NSites,i) for i in 0:2*NSites])
        zz = zip(jED.N_el.(s.states), jED.S.(s.states)) 
        @test length(s.blocklist) == length(unique(zz))
    end
end

@testset "Filter Basis" begin
    @test_throws ArgumentError jED.Basis(2, N_filter=[-2], S_filter=[]) 
    @test all(jED.N_el.(jED.Basis(2, N_filter=[0], S_filter=[]).states) .== 0)
    @test all(jED.S.(jED.Basis(2, N_filter=[], S_filter=[0]).states) .== 0)
end

@testset "Internals" begin
    @test all(jED._block_slice((4,6,1,1)) .== 4:9)
end
