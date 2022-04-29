#!/bin/bash

DATA_SET=water_GMX50_bare
wget -c https://ftp.gromacs.org/pub/benchmarks/${DATA_SET}.tar.gz
tar xf ${DATA_SET}.tar.gz
cd ./water-cut1.0_GMX50_bare/1536

export GROMACS_TAG=2020.2

singularity build ${GROMACS_TAG}.sif docker://nvcr.io/hpc/gromacs:${GROMACS_TAG}

#SIMG=${GROMACS_TAG}.sif

#SINGULARITY="singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd ${SIMG}"

#${SINGULARITY} gmx grompp -f pme.mdp

#${SINGULARITY} gmx mdrun -ntmpi 4 -nb gpu -pin on -v -noconfout -nsteps 5000 -ntomp 10 -s topol.tpr

