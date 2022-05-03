#!/bin/bash
#SBATCH --job-name=LSTM_A100_DATA
#SBATCH --output=%x.%j.o
#SBATCH --error=%x.%j.e
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar

module load gcc cuda cudnn
. $HOME/conda/etc/profile.d/conda.sh
conda activate tensorflow
./clean # remove any results from prior runs and create a results folder
./launch
