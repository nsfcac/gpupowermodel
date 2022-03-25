#!/bin/bash

echo " "
echo " ============================ Start of DEMO ============================ "

gcc apps/demo/hello_world.c -o apps/demo/hello_world.out
./apps/demo/hello_world.out | tee results/DEMO_RUN_OUT

echo " ============================= END OF DEMO ============================= "
echo " "
