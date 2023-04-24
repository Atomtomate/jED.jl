NB = 1
NF = 1
NV = length(es.evals)
ωGrid = 2 .* (-NB:NB) .* π ./ β
νGrid = 2 .* (-NF:NF-1) .* π ./ β
νpGrid = 2 .* (-NF:NF-1) .* π ./ β

νShape = (1,length(νGrid))
νpShape = (1,1,length(νGrid))
MShape = (1,1,1,NV)
NShape = (1,1,1,1,NV)
OShape = (1,1,1,1,1,NV)
PShape = (1,1,1,1,1,1,NV)

expE = exp.(- β .* es.evals)
ev   = es.evals

t1   = 1 ./ (ωGrid .+ reshape(ev, MShape) .- reshape(ev, NShape)) 
t2   = 1 ./ (reshape(νGrid, νShape) .+ reshape(ev, NShape) .- reshape(ev, OShape))
t3_a = (reshape(expE, OShape) .+ reshape(expE, PShape)) ./ (reshape(νGrid, νpShape) .+ reshape(ev, OShape) .- reshape(ev, PShape))
t3_b = (reshape(expE, NShape) .+ reshape(expE, PShape)) ./ (reshape(νGrid, νShape) .+ reshape(νGrid, νpShape) .+ reshape(ev, NShape) .- reshape(ev, PShape))
t4   = 1 ./ (ωGrid .+ reshape(νGrid, νShape) .+ reshape(ev, MShape) .- reshape(ev, OShape))
t5   = (reshape(expE, MShape) .+ reshape(expE, PShape)) ./ (ωGrid .+ reshape(νGrid, νpShape) .+ reshape(ev, MShape) .- reshape(ev, PShape))

tt = t1 .* (t2 .* (t3_a .+ t3_b) .- t4 .* (t3_a .- t5));
