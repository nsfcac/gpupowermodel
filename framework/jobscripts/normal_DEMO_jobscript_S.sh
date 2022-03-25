#!/bin/bash

app=demo

#SBATCH --job-name=$app
#SBATCH --output=%x.o%j
#SBATCH –-error=%x.e%j
#SBATCH --partition nocona
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=3994MB ##3.9GB, Modify based on needs

./demo_app_script.sh
