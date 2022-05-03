#!/bin/bash
#SBATCH --job-name=GROMACS_A100_DATA_2
#SBATCH --output=%x.%j.o
#SBATCH --error=%x.%j.e
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1

./build_GROMACS_NGC.sh # build Singularity container and do a test run
./clean # remove any results from prior runs and create a results folder
./launch
