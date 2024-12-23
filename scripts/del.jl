using Pidfile
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED
using Optim, LsqFit


U     = parse(Float64, ARGS[1])
β     = parse(Float64, ARGS[2])
NSites= parse(Int, ARGS[3])
μin   = parse(Float64,ARGS[4])
μc    = parse(Float64, ARGS[5])
μfi   = parse(Float64, ARGS[6])
#out_path  = ARGS[7]
#ID  = ARGS[8]


function run_scan()
    μr = if  μin<μfi
        vcat(μin:+0.01:μc-0.01001,μc-0.01:+0.001:μc+0.00901,μc+0.01:+0.01:μfi+0.00001)
    else
        vcat(μin:-0.01:μc+0.01001,μc+0.01:-0.001:μc-0.00901,μc-0.01:-0.01:μfi-0.00001)
    end
    # Ur = from0 ? LinRange(0,6,26)[7:20] : reverse(LinRange(0,6,26)[7:20])
    for μi in μr
        println("$μi,$β,$U")
    end
end
run_scan()
