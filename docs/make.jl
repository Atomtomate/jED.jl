using jED
using Documenter

DocMeta.setdocmeta!(jED, :DocTestSetup, :(using jED); recursive=true)

makedocs(;
    modules=[jED],
    authors="Julian Stobbe",
    repo="https://github.com/Atomtomate/jED.jl/blob/{commit}{path}#{line}",
    sitename="jED.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Atomtomate.github.io/jED.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Atomtomate/jED.jl.git"
)
