using Distributed
@everywhere using SharedArrays
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere using jED


@everywhere function DMFT_Loop(U::Float64, μ::Float64, β::Float64, kG::jED.KGrid; Nν::Int = 500, maxit::Int = 30)
    ϵₖ = [1.86, -1.23, -0.33, 0.15]
    Vₖ = [0.21, 0.30,  0.27,  0.20]
    p  = AIMParams(ϵₖ, Vₖ)
    α  = 0.7
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing
    GLoc_i = nothing
    dens = 0.0
    basis  = jED.Basis(length(Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1))
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)

    for i in 1:maxit
        # println("    ========== ITERATION $(rpad(i,3)) ==========    ")
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis, verbose=false);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        # println("     Calculating GImp")
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=1e-12, overlap=overlap, with_density=true)
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        # println("     Calculating GLoc")
        GLoc_i = jED.GLoc(ΣImp_i, μ, νnGrid, kG)

        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        # println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        # println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        # println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2))")
    end
    return p, dens #, GImp_i, ΣImp_i
end

#p, dens = DMFT_Loop(1.0, 1.0, 1.0, kG, maxit = 1)


# Urange = collect(1.0:0.5:10.0)
# μrange = collect(0:0.02:1)
# βrange = collect(1:0.5:20)
Urange = collect(1.0:0.5:2.0)
μrange = collect(0:0.5:1)
βrange = collect(1:0.5:2)
grid = SharedVector(collect(Base.product(Urange, μrange, βrange))[:])
res_p = SharedVector{NTuple{8,Float64}}(length(grid))
res_dens = SharedVector{Float64}(length(grid))


@everywhere function run_chunk!(param_grid::AbstractVector, res_p::AbstractVector, res_dens::AbstractVector)
    my_indices = localindices(param_grid)
    Nk = 60
    kG     = jED.gen_kGrid("fcc-0.14433756729740646-0.0-0.0", Nk)
    for i in my_indices
        U,μ,β = param_grid[i]
        p, dens = DMFT_Loop(U, μ, β, kG, maxit = 1)
        res_p[i] = Tuple(vcat(p.ϵₖ,p.Vₖ))
        res_dens[i] = dens
    end
end

function DMFT_grid!(param_grid::AbstractVector, res_p::AbstractVector, res_dens::AbstractVector)
   @sync begin
       for p in procs()
           @async remotecall_wait(run_chunk!, p, param_grid, res_p, res_dens)
       end
        println("Submission of $(length(param_grid)) jobs over $(nprocs()) processors done!")
   end
end
