#!/bin/bash

#SBATCH --job-name=LAMMPS_DATA
#SBATCH --output=%x.%j.o
#SBATCH --error=%x.%j.err
#SBATCH --partition matador 
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=40
##SBATCH --mem-per-cpu=9625MB

set -euf -o pipefail

#readonly gpu_count=${1:-$(nvidia-smi --list-gpus | wc -l)}
readonly gpu_count=1
readonly input=${LMP_INPUT:-in.lj.txt}

# TODO: inject one line to set the GPU freq via Slurm

ml gcc/8.4.0 openmpi/4.0.4-cuda lammps/20200505-cuda-mpi-openmp
echo "Running Lennard Jones 8x4x8 example on ${gpu_count} GPUS..."
./init.sh ./demo.sh DEMO
