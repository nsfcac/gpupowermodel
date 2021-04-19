#!/bin/bash
##############################PYTORCH MM###########################################

echo "Load test for Pytorch Matrix Multiplication"
prec="fp64"
app="pymm"
mode="load"
pwrlimit=125
arch="P100"
scale=4
file="test_pymm/results/$app-$prec-$mode-$pwrlimit-$arch-"
oldfile="changeme"
metrixSize=(5000 10000 15000 20000 25000) 
app=pytorchmm.py
for i in $(seq 0 $scale)
do 
   pushd ..
   base=$(pwd)/
   popd
   pushd test_pymm
   appPath=$(pwd)/ 
   popd
   ../utility/control $pwrlimit $base python3 $appPath$app ${metrixSize[$i]}
   mv ../utility/$oldfile $file${metrixSize[$i]} 
   echo "Iteration->$file${metrixSize[$i]}"
done

##############################CUBLAS DGEMM###########################################
: '
echo "Load test for CUBLAS DGEMM"
prec="fp64"
app="dgemm"
mode="load"
arch="V100"
pwrlimit=250                                                                                                         scale=4
file="test_cuDGEMM/results/$app-$prec-$mode-$pwrlimit-$arch-"
oldfile="changeme"
metrixSize=(16 32 48 64 72) #V100 Arch16GB supports upto 72
app=matrixMulCUBLAS
for i in $(seq 0 $scale)
do
   pushd ..
   base=$(pwd)/
   popd
   pushd test_cuDGEMM
   appPath=$(pwd)/
   popd                                                                                                                 
   ../utility/control $pwrlimit $base $appPath$app ${metrixSize[$i]}
   
   mv ../utility/$oldfile $file${metrixSize[$i]}
   
   echo "Iteration->$file${metrixSize[$i]}"
done
'

