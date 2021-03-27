#!/bin/bash
#
#SBATCH -p debug
#SBATCH -N 8 # number of nodes
#SBATCH -n 128 # number of cores
#SBATCH -t 0-10:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

source /homes/isultan/miniconda3/bin/activate pygio
# srun -n $SLURM_NTASKS --mpi=pmi2 python cc_reduce.py
srun -n $SLURM_NTASKS --mpi=pmi2 python /homes/isultan/projects/smacc_parallel/cc_generate_parallel.py