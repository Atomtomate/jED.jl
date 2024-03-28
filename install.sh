#!/bin/bash

curl -fsSL https://install.julialang.org | sh

read -p "Enter path to which jED will be cloned: " path
git clone https://github.com/Atomtomate/jED.jl.git "$path"
cd "$path"
julia -e "using Pkg; Pkg.activate(\".\"); Pkg.instantiate()"
