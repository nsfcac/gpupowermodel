#!/bin/bash

echo " "
echo " ============================ Start of DGEMM =========================== "

./matrixMulCUBLAS 72 | tee results/DGEMM_RUN_OUT
sleep 2

echo " ============================= END OF DGEMM ============================ "
echo " "
