#!/bin/bash

numRuns=3
prec="fp64"
mode="run"
arch="P100" #V100
oldfile="changeme"

#Read supported frequencies on the platform:
nvidia-smi -i 0 -q -d SUPPORTED_CLOCKS > temp

: '
#command line benchmark specific parameters
declare -a progs=("FIRESTARTER_CUDA" "cuda-stream" "matrixMulCUBLAS") #executable name
declare -a apps=("firestarter" "stream" "dgemm") #any name
declare -a appParams=(" -t 300" " -s 655360000 -n 1500 > results/BSRUN_OUT" " 72 > results/DGEMMRUN_OUT") # ex
'
declare -a progs=("cuda-stream" "matrixMulCUBLAS") #executable name
declare -a apps=("stream" "dgemm") #any name
declare -a appParams=(" --device 1 -s 655360000 -n 1500 > results/BSRUN_OUT" " 72 > results/DGEMMRUN_OUT") # ex

nRuns=$((numRuns-1))
for i in $(seq 0 $nRuns)
do
    appID=0
    for app in "${apps[@]}"
    do
	# DVFS
	while IFS= read -r line 
	do 
	    IFS=':' read -ra l <<< "$line"
	    s=$(echo ${l[0]} | tr -d ' ')
	    if [[ $s == "Graphics" ]] 
	    then 
		IFS=' ' read -ra ll <<< ${l[1]} 

		 # GET RESULTS WITH NEW FREQUENCY

		echo "### $app - ${ll[0]} (MHz) - Iteration: $i ###"
	
		curPwrLimit=250
		freq=${ll[0]}
		file="results/$app-$prec-$mode-$curPwrLimit-$arch-$freq-"
		
		pushd ..
		base=$(pwd)/
		popd
		pushd apps
		appPath=$(pwd)/
		popd
                
		appCmd="$curPwrLimit $freq $base $appPath${progs[$appID]}${appParams[$appID]}"
		#echo $appCmd
		
		../utility/control $appCmd
		mv ../utility/$oldfile $file$i
		
		# SAVE HBM BW for STREAM
		if [[ $app == "stream" ]]
		then
		    r=$(grep Triad results/BSRUN_OUT)
		    echo $r | cut -d" " -f 2 >> results/run_triads_dvfs_$freq
		fi
		
		# SAVE GFLOPS for DGEMM
		if [[ $app == "dgemm" ]]
                then                                                                                             
                    r=$(grep Performance results/DGEMMRUN_OUT)
		    echo $r
		    IFS=',' read -ra DGEMM <<< "$r"
		    IFS=' ' read -ra FLOPS <<< "${DGEMM[0]}"
		    IFS=' ' read -ra TIMEMS <<< "${DGEMM[1]}"
		    echo "${FLOPS[1]} ${TIMEMS[1]}" >> results/run_perf_dvfs_$freq
		fi
		
	    fi 
	done <"temp" # loop through supported frequency ranges
	
	# POWER Cap: 125 to 250 @ 5W increment
	curPwrLimit=125 # read from the supported lower power limits
	while [ $curPwrLimit -le 250 ] # read from the supported upper power limits
	do
	    # GET RESULTS WITH NEW POWER LIMIT
	    echo "### $app - $curPwrLimit (W) - Iteration: $i ###"

	    freq=1380 # read max from the supported clocks
	    file="results/$app-$prec-$mode-$curPwrLimit-$arch-$freq-"
	    
            pushd ..
            base=$(pwd)/
            popd
            pushd apps
            appPath=$(pwd)/
            popd

            appCmd="$curPwrLimit $freq $base $appPath${progs[$appID]}${appParams[$appID]}"
            #echo $appCmd
            
	    ../utility/control $appCmd
	    mv ../utility/$oldfile $file$i
	    
	    #results/BSRUN_OUT_$i

	    if [[ $app == "dgemm" ]]
                then
                    r=$(grep Performance results/DGEMMRUN_OUT)
                    echo $r
                    IFS=',' read -ra DGEMM <<< "$r"
                    IFS=' ' read -ra FLOPS <<< "${DGEMM[0]}"
                    IFS=' ' read -ra TIMEMS <<< "${DGEMM[1]}"
		    echo "${FLOPS[1]} ${TIMEMS[1]}" >> results/run_perf_pl_$curPwrLimit
            fi
	    
	    # SAVE HBM BW for STREAM
            if [[ $app == "stream" ]]
            then
                r=$(grep Triad results/BSRUN_OUT)
                echo $r | cut -d" " -f 2 >> results/run_triads_pl_$curPwrLimit
	    fi
	    
	    #update the power limit
	    curPwrLimit=$((curPwrLimit+5))
	done
	appID=$((appID+1))
   done #apps

done #runs


: '
echo "####################################################FIRESTARTER#########################################"
prec="fp64"
app="firestarter"
mode="run"
arch="P100"
pwrlimit=250
numRuns=11
pstate=544
file="test_FIRESTARTER/results/$app-$prec-$mode-$pwrlimit-$arch-$pstate-"
oldfile="changeme"
prog=FIRESTARTER_CUDA
for i in $(seq 0 $numRuns)
do
   pushd ..
   base=$(pwd)/
   popd
   pushd test_FIRESTARTER
   appPath=$(pwd)/
   popd
   ../utility/control $pwrlimit $pstate $base $appPath$prog -t 300
   mv ../utility/$oldfile $file$i
   echo "Run->$app$i"
done

echo "#####################################################BABEL STREAM#########################################"
app="stream"
prec="fp64"
mode="run"
arch="P100"
pwrlimit=125
numRuns=11
pstate=P0
file="test_STREAM/results/$app-$prec-$mode-$pwrlimit-$arch-$pstate-"
oldfile="changeme"
prog=cuda-stream

for i in $(seq 0 $numRuns)
do
   pushd ..
   base=$(pwd)/
   popd
   pushd test_STREAM
   appPath=$(pwd)/
   popd
   ../utility/control $pwrlimit $pstate $base $appPath$prog -s 655360000 -n 1500 > $appPath/results/BSRUN_OUT_$i
   mv ../utility/$oldfile $file$i
   r=$(grep Triad $appPath/results/BSRUN_OUT_$i)
   echo $r | cut -d" " -f 2 >> $appPath/results/runtriads_$pwrlimit
   echo "Run->$app$i"
done


##############################CUBLAS DGEMM###########################################

prec="fp64"
app="dgemm"
mode="run"
arch="P100"
pwrlimit=250
numRuns=11
pstate=544
file="test_cuDGEMM/results/$app-$prec-$mode-$pwrlimit-$arch-$pstate-"
oldfile="changeme"
prog=matrixMulCUBLAS
for i in $(seq 0 $numRuns)
do
   pushd ..
   base=$(pwd)/
   popd
   pushd test_cuDGEMM
   appPath=$(pwd)/
   popd
   ../utility/control $pwrlimit $pstate $base $appPath$prog 72 > $appPath/results/DGEMMRUN_OUT_$i
   mv ../utility/$oldfile $file$i

   r=$(grep Performance $appPath/results/DGEMMRUN_OUT_$i)
   echo $r
   IFS=',' read -ra DGEMM <<< "$r"
   IFS=' ' read -ra FLOPS <<< "${DGEMM[0]}"
   IFS=' ' read -ra TIMEMS <<< "${DGEMM[1]}"
   echo "${FLOPS[1]} ${TIMEMS[1]}" >> $appPath/results/runperf_$pstate

   echo "Run->$app$i"
done
'
