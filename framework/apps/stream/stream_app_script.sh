#!/bin/bash

echo " "
echo " =========================== Start of STREAM =========================== "

./cuda-stream --device 0 -s 655360000 --triad-only -n 300 | tee results/STREAM_RUN_OUT
sleep 2

echo " ============================ END OF STREAM ============================ "
echo " "
