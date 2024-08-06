#! /bin/bash
#SBATCH -J gen_batch1
##SBATCH -o pyt_cli_test_conv_gpu.out
#SBATCH --time=01:00:00
#SBATCH --partition=standard96:test
#SBATCH --nodes=1

julia -p24 script_MLDataset.jl
