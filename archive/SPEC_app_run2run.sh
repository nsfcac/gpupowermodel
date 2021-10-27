#!/bin/bash

numRuns=3
prec="fp64"
mode="run"
arch="P100" #V100
oldfile="changeme"

#Read supported frequencies on the platform:
temp=$(nvidia-smi -i 0 -q -d SUPPORTED_CLOCKS)
declare -a benchmarks=("tpacf" "stencil" "lbm" "fft" "spmv" "mriq" "histo" "bfs" "cutcp" "kmeans" "lavamd" "cfd" "nw" "hotspot" "lud" "ge" "srad" "heartwall" "bplustree")

declare -a benchmarks=("tpacf")

nRuns=$((numRuns-1))

  # benchmarks   
  for bm in "${benchmarks[@]}"
  do
 	# DVFS
  	while IFS= read -r line 
	do 	
	    IFS=':' read -ra l <<< "$line"
	    s=$(echo ${l[0]} | tr -d ' ')
	    if [[ $s == "Graphics" ]] 
	    then 
		for i in $(seq 0 $nRuns)
		do
			IFS=' ' read -ra ll <<< ${l[1]} 

		 	# GET RESULTS WITH NEW FREQUENCY

			echo "### ${ll[0]} (MHz)-->Run =$i ###"
	
			curPwrLimit=250
			freq=${ll[0]}
			file="result/$prec-$mode-$curPwrLimit-$arch-$bm-$freq-"
		
			pushd ..
			base=$(pwd)/
			popd
			#pushd apps
			#appPath=$(pwd)/
			#popd
            runCmd="runspec --config=opencl-nvidia-simple.cfg --platform NVIDIA --iterations=1 --device GPU --size=ref --noreportable --tune=base --output_format csv $bm -I" 
			appCmd="$curPwrLimit $freq $base $runCmd"
			echo $appCmd
			../utility/control $appCmd
			mv ../utility/$oldfile $file$i
		done # n runs
	   fi
  done <<< "$temp" # loop through supported frequency ranges
done #benchmarks
	: '
	# POWER Cap: 125 to 250 @ 5W increment
	curPwrLimit=125 # read from the supported lower power limits
	while [ $curPwrLimit -le 250 ] # read from the supported upper power limits
	do
	    # GET RESULTS WITH NEW POWER LIMIT
	    echo "### $curPwrLimit (W) ###"

	    freq=1328 # read max from the supported clocks
	    file="results/$prec-$mode-$curPwrLimit-$arch-$freq-"
	    
            pushd ..
            base=$(pwd)/
            popd
            #pushd apps
            #appPath=$(pwd)/
            #popd

            appCmd="$curPwrLimit $freq $base $runCmd"
            echo $appCmd
            
	    ../utility/control $appCmd
	    mv ../utility/$oldfile $file$i
	    
	    #update the power limit
	    curPwrLimit=$((curPwrLimit+5))
	done
	'

