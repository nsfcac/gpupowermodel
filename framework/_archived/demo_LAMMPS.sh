#!/bin/bash
echo " =========================== Start of DEMO =========================== "

declare -a appOutput=(" > results/LAMMPS_RUN_OUT") 
appID=0

readonly gpu_count=1
readonly input=${LMP_INPUT:-in.lj.txt}
mpirun -n ${gpu_count} lmp -k on g ${gpu_count} -sf kk -pk kokkos cuda/aware on neigh full comm device binsize 2.8 -var x 8 -var y 4 -var z 8 -in ${input}${appOutput[$appID]}

# appCmd="mpirun -n ${gpu_count} lmp -k on g ${gpu_count} -sf kk -pk kokkos cuda/aware on neigh full comm device binsize 2.8 -var x 8 -var y 4 -var z 8 -in ${input}${appOutput[$appID]}"
# echo $appCmd

echo " ============================ END OF DEMO ============================ "
