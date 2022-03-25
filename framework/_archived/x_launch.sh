#!/bin/bash
echo "Welcome to Optimal GPU Frequency Selection Framework!"

numberOfArgs=$#
echo "The total number of arguments is $numberOfArgs"

declare -a execs=("./demo") #executable name
declare -a apps=("DEMO") #any name

if [[ $numberOfArgs < 2 ]]
then
  echo "Please provide arguments: "
  echo "  [1] executable name"
  echo "  [2] application name"
elif [[ $numberOfArgs == 2 ]]
then
  declare -a execs=($1) #executable name
  echo "First arg: ${execs[0]}"

  declare -a apps=($2) #any name
  echo "Second arg: ${apps[0]}"
else
  declare -a execs=($1) #executable name
  echo "First arg: ${execs[0]}"

  declare -a apps=($2) #any name
  echo "Second arg: ${apps[0]}"

  declare -a appParams=($3) 
  echo "Third arg: ${appParams[0]}"
fi

declare -a appParams=(" > results/${apps[0]}_RUN_OUT") 

# =============================================================================

numRuns=1
mode="dvfs"
arch="GV100" #GV100
oldfile="changeme"
m_freq=877
sleep_interval=10
t=0

# declare -a freqs=(1380 1372 1365 1357 1350 1342 1335 1327 1320 1312 1305 1297 \
#                   1290 1282 1275 1267 1260 1252 1245 1237 1230 1222 1215 1207 \
#                   1200 1192 1185 1177 1170 1162 1155 1147 1140 1132 1125 1117 \
#                   1110 1102 1095 1087 1080 1072 1065 1057 1050 1042 1035 1027 \
#                   1020 1012 1005 997 990 982 975 967 960 952 945 937 930 922 \
#                   915 907 900 892 885 877 870 862 855 847 840 832 825 817 810 \
#                   802 795 787 780 772 765 757 750 742 735 727 720 712 705 697 \
#                   690 682 675 667 660 652 645 637 630 622 615 607 600 592 585 \
#                   577 570 562 555 547 540 532 525 517 510 502 495 487 480 472 \
#                   465 457 450 442 435 427 420 412 405)
# declare -a freqs=(1380 547)
declare -a freqs=(1380)

for c_freq in "${freqs[@]}"
do
  ./control $m_freq $c_freq
  nRuns=$((numRuns-1))
  for i in $(seq 0 $nRuns)
  do
    appID=0
    for app in "${apps[@]}"
      do  
      t=$((t+1))
      # GET RESULTS WITH NEW FREQUENCY
      echo "### $app - $c_freq (MHz) - Iteration: $i ###"
      file="results/$arch-$mode-$app-$c_freq-$i"
      
      #pushd apps
      #appPath=$(pwd)/
      #popd
          
      #appCmd="$appPath${execs[$appID]}${appParams[$appID]}"
      #readonly gpu_count=${1:-$(nvidia-smi --list-gpus | wc -l)}
      
      # Application Executable for LAMMPS =====================================
      # readonly gpu_count=1
      # readonly input=${LMP_INPUT:-in.lj.txt}
      # appCmd="mpirun -n ${gpu_count} lmp -k on g ${gpu_count} -sf kk -pk \
      #         kokkos cuda/aware on neigh full comm device binsize 2.8 -var \
      #         x 8 -var y 4 -var z 8 -in ${input}${appParams[$appID]}"
      # =======================================================================

      appCmd=${execs[0]}
      echo $appCmd
      ./profile.py $appCmd

      cp $oldfile $file
      rm -f $oldfile
      sleep $sleep_interval

      # SAVE time for LAMMPS ==================================================
      if [[ $app == "LAMMPS" ]]
      then
        v=$(sed -n '62p' results/LAMMPS_RUN_OUT)
        tokens=( $v )
        wall_time=${tokens[3]}
        echo $wall_time
        wall_time="${wall_time//[$'\t\r\n ']}"
        printf '%s\n' $c_freq $wall_time | paste -sd ',' >> results/$arch-dvfs-lammps-perf.csv
      fi

			# SAVE time for LSTM ====================================================
			if [[ $app == "LSTM" ]]
			then
				v=$(sed -n '29p' results/LSTM_RUN_OUT)
				tokens=( $v )
				exec_time=${tokens[0]}
				echo $exec_time
				exec_time="${exec_time//[$'\t\r\n ']}"
				printf '%s\n' $c_freq $exec_time | paste -sd ',' >> results/$arch-dvfs-lstm-perf.csv
			fi

      # SAVE time for NAMD ====================================================
			if [[ $app == "NAMD" ]]
			then
				#v=$(sed -n '258p' results/NAMD_RUN_OUT)
				v=$(grep "WallClock:" results/NAMD_RUN_OUT)
        tokens=( $v )
				wall_time=${tokens[1]}
				echo $wall_time
				wall_time="${wall_time//[$'\t\r\n ']}"
				printf '%s\n' $c_freq $wall_time | paste -sd ',' >> results/$arch-dvfs-namd-perf.csv
			fi

      appID=$((appID+1))
    done #apps
  done #runs
done #freqs

echo "DONE!!!"

# revert the core frequency
c_freq=1380
./control $m_freq $c_freq
echo $t


# #!/bin/bash
# START=$(date +%s)
# # do something

# # start your script work here
# ls -R /etc > /tmp/x
# rm -f /tmp/x
# # your logic ends here

# END=$(date +%s)
# DIFF=$(( $END - $START ))
# echo "It took $DIFF seconds"