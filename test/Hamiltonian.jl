@testset "overlap" begin
    t1 = jED.SVector{8}(Bool[1,1,1,1,0,1,0,1])
    t2 = jED.SVector{8}(Bool[1,1,0,1,1,1,0,1])
    t3 = jED.SVector{8}(Bool[1,1,0,1,1,1,1,1])
    test1 = false
    #t1 and t3 should never have an overlap
    for i in 1:8
        for j in 1:8
            test1 = test1 | (jED.overlap_cdagger_c(t1,t3,i,j) != 0)
        end
    end

    @test !test1
    @test jED.overlap_cdagger_c(t1,t2,3,3) == 0
    @test jED.overlap_cdagger_c(t1,t1,2,2) == 1
    @test jED.overlap_cdagger_c(t1,t1,5,5) == 0
    @test jED.overlap_cdagger_c(t1,t2,3,5) == -1
end
