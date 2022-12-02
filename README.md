# jED

Simple implementation of exact diagonalization for models such as die Hubbard or Anderson impurity model (AIM).
Defines overlap and operators which can be used to build up a model. Examples for the single impurity AIM are given in (TODO: name of file)

Most of this code is adapted from [sfED.jl](https://github.com/steffenbackes/sfED) with some minor changes to improve performance for simple models. 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Atomtomate.github.io/jED.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Atomtomate.github.io/jED.jl/dev/)
[![Build Status](https://github.com/Atomtomate/jED.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/Atomtomate/jED.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/Atomtomate/jED.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Atomtomate/jED.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

## Installation

  - install [Julia](https://julialang.org/downloads) (1.7 +)
    - download tar
    - extract to arbitrary directory
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
