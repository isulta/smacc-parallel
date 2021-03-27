#!/bin/bash
#
#SBATCH -p debug
#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-00:05 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

source /homes/isultan/miniconda3/bin/activate pygio
# srun -n $SLURM_NTASKS --mpi=pmi2 python int1dtest.py
# srun -n $SLURM_NTASKS --mpi=pmi2 python m21test.py
srun -n $SLURM_NTASKS --mpi=pmi2 python h5test.py