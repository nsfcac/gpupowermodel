#!/bin/bash

# wget https://lammps.sandia.gov/inputs/in.lj.txt
# export BENCHMARK_DIR=$PWD

# cd $BENCHMARK_DIR
# wget https://gitlab.com/NVHPC/ngc-examples/-/raw/master/lammps/single-node/run_lammps.sh
# chmod +x run_lammps.sh

export LAMMPS_TAG=29Oct2020
#29Sep2021

singularity build ${LAMMPS_TAG}.sif docker://nvcr.io/hpc/lammps:${LAMMPS_TAG}

SINGULARITY="$(which singularity) run --nv -B $(pwd):/host_pwd --pwd /host_pwd ${LAMMPS_TAG}.sif"

${SINGULARITY} ./run_lammps.sh

#singularity run --nv -B $PWD:/host_pwd --pwd /host_pwd docker://nvcr.io/hpc/lammps:${LAMMPS_TAG} ./run_lammps.sh

