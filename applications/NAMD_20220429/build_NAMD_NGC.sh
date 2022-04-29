#!/bin/bash

wget -O - https://gitlab.com/NVHPC/ngc-examples/raw/master/namd/3.0/get_apoa1.sh | bash
cd ./apoa1

export NAMD_TAG=3.0-alpha9-singlenode 
export NAMD_EXE=namd3

singularity build ${NAMD_TAG}.sif docker://nvcr.io/hpc/namd:${NAMD_TAG}

SINGULARITY="$(which singularity) run --nv -B $(pwd):/host_pwd --pwd /host_pwd ${NAMD_TAG}.sif"

${SINGULARITY} ${NAMD_EXE} +ppn $(nproc) +setcpuaffinity +idlepoll $(pwd)/apoa1_nve_cuda_soa.namd
