#!/bin/bash
#SBATCH --job-name=LAMMPS_A100_DATA
#SBATCH --output=%x.%j.o
#SBATCH --error=%x.%j.e
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar

module load gcc cuda
./launch
