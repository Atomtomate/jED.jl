# jED

|     Build Status    |      Coverage      |  Documentation |    Code Style  |
| ------------------- |:------------------:| :-------------:| :-------------:|
| [![Build Status](https://github.com/Atomtomate/jED.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/Atomtomate/jED.jl/actions/workflows/CI.yml?query=branch%3Amaster) | [![Coverage](https://codecov.io/gh/Atomtomate/jED.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Atomtomate/jED.jl) | [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Atomtomate.github.io/jED.jl/dev/) [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Atomtomate.github.io/jED.jl/stable/) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) |

------------------------------------

Simple implementation of exact diagonalization for models such as die Hubbard or Anderson impurity model (AIM).
Defines overlap and operators which can be used to build up a model. Examples for the single impurity AIM are given in (TODO: name of file)

The original version of this code was adapted from [sfED.jl](https://github.com/steffenbackes/sfED) with some minor changes to improve performance for simple models. 

## Installation

  - install [Julia](https://julialang.org/downloads) (1.7 +)
    - download tarextract to an arbitrary directory
    - put executable (located in 'bin' subdirectory of tar) into PATH variable  
  - install jED
    - `git clone https://github.com/Atomtomate/jED.jl .`
    - `cd jED`
    - run Julia package manager by running `julia` and pressing `]`
    - activate project by typing `activate .`
    - install dependencies by typing `instantiate`

## Example run

The file `example1.jl` contains code for the execution of a simple Anderson impurity model calculation.
In order to run the example follow these steps:

  - cd into jED directory
  - run Julia
  - run `include("example1.jl")`

You can now also inspect all quantities, such as the basis, model, eigenspace and so on.
Further documentation is available [[here]](https://Atomtomate.github.io/jED.jl/dev/).

## High precision and special models

There are multiple examples in the scripts directory, showing the use of different models (such as the Hofstadter model) and the use of higher precision.
The second use case may improve stability for analytic continuation, such as [[Nevanlinna]](https://github.com/SpM-lab/Nevanlinna.jl).

## Anderson parameters fitting procedure

Since the fitting of the parameters of the finite bath for the Anderson impurity model (the only one implemented for now) can be tricky, there are multiple examples in the `scripts` directory exploring different options.

There are two different example scripts for parameter fitting: 
    
- In `scripts/example3` and `script/example4` you can find examples of tests with different cost functions

- In `scripts/example16` you can find an example for a global optimum search based on [`NLopt`](https://nlopt.readthedocs.io/en/latest/). This can be especially useful for the first iterations of calculations with a high number of bath sites.


## Fortran compatibility

The `fortran_compat.jl` script is a headless (HPC compatible) script, usable as a drop-in replacement for a well-known Fortran77 ED code.

Usage is explained as comments, it can be called from bash (e.g. slurm scripts) with:
```
/PATH/TO/JULIA/BIN/julia /PATH/TO/SCRIPT_DIR/fortran_compat.jl 1.1 1.2 1.3 10 3dsc-1.1-1.2-1.3 /PATH/TO/OUTPUT_DIR
```

Where `U = 1.1`, `beta = 1.2`, `mu = 1.3`, `Nk = 10` and lattice type 3D simple cubic with `t=1.1`, `t'=1.2`, `t''=1.3`.
See [[Dispersions.jl]](https://github.com/Atomtomate/Dispersions.jl) for available tight binding models!

For now, the number of Fermionic frequencies, maximum number of iterations and convergence epsilon are 'hardcoded' inside the script. 
If necessary, those can be either changed in place or made available to the bash script as parameters, by parsing more `ARGS` parameters from the command line.

## Performace

This code is written with readability as the main goal to demonstrate the ED algorithm.
However, if you decide to run this code for larger scale models, see `example15` in the `scripts` directory for hints to improve performance slightly.

## Dependencies

Some examples use the [[SparseIR]](https://github.com/SpM-lab/SparseIR.jl) package. Please read their LICENSE file before adopting any code from this repository using the sparseIR basis package!
