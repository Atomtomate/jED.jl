@testset "operator ni" begin
    t1 = jED.SVector{8}(Bool[1,1,1,1,0,1,0,1])
    r1 = jED.operator_ni(t1, 1)
    r2 = jED.operator_ni(t1, 5)
    @test r1[1] == 1
    @test all(r1[2] .== t1)
    @test r2[1] == 0
    @test all(r2[2] .== t1)
end

@testset "operator c/cdag" begin
    t1 = jED.SVector{8}(Bool[1,1,1,1,0,1,0,1])
    t2 = jED.SVector{8}(Bool[0,1,1,1,0,1,0,1])
    t3 = jED.SVector{6}(Bool[1,0,1,0,0,1])
    t4 = jED.SVector{6}(Bool[1,0,1,1,0,1])
    c1 = create_op(basis, 4)
    c2 = ann_op(basis, 4)
    c3 = create_op(basis, 1)
    c4 = ann_op(basis, 1)
    @test c1.N_inc == 1
    @test c2.N_inc == -1
    @test c1.S_inc == -1
    @test c2.S_inc == 1
    @test c1.N_inc == 1
    @test c2.N_inc == -1
    @test c1.S_inc == -1
    @test c2.S_inc == 1
    @test all(c1(t3) .== t4)
    @test all(c2(t4) .== t3)
    @test jED.create(t1, 1) === nothing
    @test all(jED.ann(t1, 1) .== t2)
    @test all(jED.create(t2, 1) .== t1)
    @test jED.ann(t2, 1) === nothing
end

@testset "overlap" begin
    t1 = jED.SVector{8}(Bool[1,1,1,1,0,1,0,1])
    t2 = jED.SVector{8}(Bool[1,1,0,1,1,1,0,1])
    @test jED.overlap(t1,t1) == 1
    @test jED.overlap(t1,t2) == 0
    op_up = create_op(basis, 1) # Creation operator for ↑ at impurity
    ov_i = jED._find_cdag_overlap_blocks(basis.blocklist, op_up)
    ov_i_fortran = [3,5,6,8,9,10,11,12,13,0,14,15,0,16,0,0] # exported from idmat
    @test all(ov_i .== ov_i_fortran) 
end

@testset "overlap cdagger c" begin
    t1 = jED.SVector{8}(Bool[1,1,1,1,0,1,0,1])
    t2 = jED.SVector{8}(Bool[1,1,0,1,1,1,0,1])
    t3 = jED.SVector{8}(Bool[1,1,0,1,1,1,1,1])
    t4 = jED.SVector{6}(Bool[1,1,1,1,0,1])
    t5 = jED.SVector{6}(Bool[1,1,1,1,1,0])
    test1 = false
    #t1 and t3 should never have an overlap
    for i in 1:8
        for j in 1:8
            test1 = test1 | (jED.overlap_cdagger_c(t1,i,t3,j) != 0)
        end
    end

    @test !test1
    @test jED.overlap_cdagger_c(t1,3,t2,3) == 0
    @test jED.overlap_cdagger_c(t1,2,t1,2) == 1
    @test jED.overlap_cdagger_c(t1,5,t1,5) == 0
    @test jED.overlap_cdagger_c(t1,3,t2,5) == -1
    @test jED.overlap_cdagger_c(t2,5,t1,3) == -1
    test2 = true
    for i in 1:6
        for j in 1:6
            res = jED.overlap_cdagger_c(t4,i,t5,j)
            if i == 6 && j == 5
                test2 = test2 & (res == 1)
            else
                test2 = test2 & (res == 0)
            end
        end
    end
    @test test2
end

@testset "overlap n_i n_j" begin
    t1 = jED.SVector{8}(Bool[1,1,1,1,0,1,0,1])
    t2 = jED.SVector{8}(Bool[1,1,0,1,1,1,0,1])
    @test jED.overlap_ni_nj(t1,t2, 1,2) == 0
    @test jED.overlap_ni_nj(t1,t1, 1,2) == 1
    @test jED.overlap_ni_nj(t1,t2, 1,5) == 0
    @test jED.overlap_ni_nj(t1,t2, 7,5) == 0
end

@testset "overlap cdag" begin
    c1 = create_op(basis, 4)
    c2 = create_op(basis, 1)
    ov_bi = jED._find_cdag_overlap_blocks(basis.blocklist, c1)
    ov_bi2 = jED._find_cdag_overlap_blocks(basis.blocklist, c2)
    for (i,bl_i) in enumerate(ov_bi) 
        if bl_i > 0 
            @test basis.blocklist[i][3] == basis.blocklist[bl_i][3] - 1
            @test basis.blocklist[i][4] == basis.blocklist[bl_i][4] + 1
            if basis.blocklist[i][4] != basis.blocklist[bl_i][4] + 1
                println("$i, $bl_i")
            end
        end
    end
    for (i,bl_i2) in enumerate(ov_bi2) 
        if bl_i2 > 0
            @test basis.blocklist[i][3] == basis.blocklist[bl_i2][3] - 1
            @test basis.blocklist[i][4] == basis.blocklist[bl_i2][4] - 1
        end
    end
end
