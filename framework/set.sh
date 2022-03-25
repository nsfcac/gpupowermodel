#!/bin/bash 

# echo " *** Starting Set Frequency *** "

m_freq=$1
c_freq=$2
echo "m_freq c_freq"
echo $m_freq $c_freq

# dcgmi config --set -a $m_freq,$c_freq
srun --gpu-freq="$c_freq"
# TODO: set job script via slurm...
# ...open jobscript and add that line setting gpu frequency to the jobscript

sleep 2

echo "*** set.sh over ***"
# echo " *** Exiting Set Frequency *** "
