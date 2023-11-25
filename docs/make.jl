using Documenter
using Pkg
Pkg.activate(String(@__DIR__) * "/..")
using jED
push!(LOAD_PATH, "../src")

#DocMeta.setdocmeta!(jED, :DocTestSetup, :(using jED); recursive=true)
makedocs(;
    modules = [jED],
    authors = "Julian Stobbe",
    repo = "https://github.com/Atomtomate/jED.jl/blob/{commit}{path}#{line}",
    sitename = "jED.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://Atomtomate.github.io/jED.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
    warnonly = Documenter.except(),
)

deploydocs(;
    branch = "gh-pages",
    devbranch = "master",
    devurl = "stable",
    repo = "github.com/Atomtomate/jED.jl.git",
)
