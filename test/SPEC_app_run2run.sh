#!/bin/bash

numRuns=1
prec="fp64"
mode="run"
arch="P100" #V100
oldfile="changeme"

#Read supported frequencies on the platform:
nvidia-smi -i 0 -q -d SUPPORTED_CLOCKS > temp

# SPEC Benchmark run command
declare -a appCmd=("runspec --config=opencl-nvidia-simple.cfg --platform NVIDIA --device GPU --tune=all --output_format csv --iterations=3 opencl ^histo")

nRuns=$((numRuns-1))
for i in $(seq 0 $nRuns)
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

		echo "### ${ll[0]} (MHz) ###"
	
		curPwrLimit=250
		freq=${ll[0]}
		file="results/$prec-$mode-$curPwrLimit-$arch-$freq-"
		
		#pushd ..
		#base=$(pwd)/
		#popd
		#pushd apps
		#appPath=$(pwd)/
		#popd
                
		#appCmd="$curPwrLimit $freq $base $appPath${progs[$appID]}${appParams[$appID]}"
		#echo $appCmd
		
		../utility/control $appCmd
		mv ../utility/$oldfile $file$i
		
	done <"temp" # loop through supported frequency ranges
	
	# POWER Cap: 125 to 250 @ 5W increment
	curPwrLimit=125 # read from the supported lower power limits
	while [ $curPwrLimit -le 250 ] # read from the supported upper power limits
	do
	    # GET RESULTS WITH NEW POWER LIMIT
	    echo "### $curPwrLimit (W) ###"

	    freq=1380 # read max from the supported clocks
	    file="results/$prec-$mode-$curPwrLimit-$arch-$freq-"
	    
            #pushd ..
            #base=$(pwd)/
            #popd
            #pushd apps
            #appPath=$(pwd)/
            #popd

            #appCmd="$curPwrLimit $freq $base $appPath${progs[$appID]}${appParams[$appID]}"
            #echo $appCmd
            
	    ../utility/control $appCmd
	    mv ../utility/$oldfile $file$i
	    
	    #update the power limit
	    curPwrLimit=$((curPwrLimit+5))
	done
done #runs
