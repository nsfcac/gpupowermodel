#!/bin/bash
echo " "
echo " ======== Welcome to Optimal GPU Frequency Selection Framework! ======== "

declare -a appScripts=("./demo.sh") # application script
declare -a appNames=("DEMO")        # application name

arch="GV100" # GV100
mode="dvfs"

oldfile="changeme"
sleep_interval=5

numRuns=1              
m_freq=877                           # memory frequency
c_freq=1380                          # core frequency

# =============================================================================
numberOfArgs=$#
echo "The total number of arguments is $numberOfArgs"

if [[ $numberOfArgs < 2 ]]
then
  echo "Please provide arguments: "
  echo "  [1] application script"
  echo "  [2] application name"
  echo "  [3] number of runs (optional)"
  echo "  [4] core frequency (optional)"
  echo "  [5] memory frequency (optional)"
elif [[ $numberOfArgs == 2 ]]
then
  declare -a appScripts=($1)         # application script
  echo "First arg: ${appScripts[0]}" 

  declare -a appNames=($2)           # application name
  echo "Second arg: ${appNames[0]}"  
else
  declare -a appScripts=($1)         # application script
  echo "First arg: ${appScripts[0]}" 

  declare -a appNames=($2)           # application name
  echo "Second arg: ${appNames[0]}"

  numRuns=$3                         # number of runs
  echo "Fifth arg: $numRuns"

  c_freq=$4                          # core frequency
  echo "Third arg: $c_freq"

  m_freq=$5                          # memory frequency
  echo "Fourth arg: $m_freq"
fi
echo " "

# =============================================================================

# echo "...Calling clean.sh for results folder"
# sleep 1
# echo "...3..."
# sleep 1
# echo "...2..."
# sleep 1
# echo "...1..."
# sleep 1
./clean.sh
# declare -a appParams=(" > results/${appNames[0]}_RUN_OUT") 

# =============================================================================

# echo "...Calling set.sh for setting frequency"
# ./set.sh $m_freq $c_freq

nRuns=$((numRuns-1))
for i in $(seq 0 $nRuns)
do
  app=${appNames[0]}
  appCmd=${appScripts[0]}

  # echo "### $app - $c_freq (MHz) - Iteration: $i ###"
  timeFile="results/$arch-$mode-$app-perf.csv"
  profileFile="results/$arch-$mode-$app-$c_freq-$i"

  echo " "
  C_OUTPUT=`python3 check.py "$app"`
  if [[ "${C_OUTPUT[0]}" -gt 0 ]]
  then
    echo "...The job will run with the optimal frequency of ${C_OUTPUT[0]}."
    # echo "...Calling set.sh for setting optimal frequency"
    
    c_freq=${C_OUTPUT[0]}
    ./set.sh $m_freq $c_freq
    
    echo " "
    echo "...running application..."
    START=$(date +%s)
    $appCmd
    END=$(date +%s)

    DIFF=$(( $END - $START ))
    echo "...Execution of application script took $DIFF seconds."
    # save execution time 
    wall_T="${DIFF//[$'\t\r\n ']}" 
    # echo $wall_T
    printf '%s\n' $c_freq $wall_T | paste -sd ',' >> $timeFile
  else  
    echo "...Your job's optimal frequency is being computed."
    # echo "...Calling set.sh for setting to default frequency"
    
    c_freq=1380
    ./set.sh $m_freq $c_freq
    
    echo " "
    echo "...running application while profiling..."
    START=$(date +%s)
    python3 profile.py $appCmd
    END=$(date +%s)

    DIFF=$(( $END - $START ))
    echo "...Execution of application script took $DIFF seconds."
    # save execution time 
    wall_T="${DIFF//[$'\t\r\n ']}" 
    # echo $wall_T
    printf '%s\n' $c_freq $wall_T | paste -sd ',' >> $timeFile

    cp $oldfile $profileFile
    rm -f $oldfile

    # TODO: find the optimal frequecy save it to database...
    M_OUTPUT=`python3 model.py "$app" "$timeFile" "$profileFile" "$c_freq" "$nRuns" "database.csv"`
    # c_freq=${M_OUTPUT[0]}
    echo  ${M_OUTPUT[0]}
    # echo "...Optimal frequency for this app is saved as ${M_OUTPUT[0]} MHz."
  fi
  echo "### RAN: $app at $c_freq (MHz) for $i iterations. ###"
  sleep $sleep_interval
done #runs

# =============================================================================
# revert the core frequency
# echo "...Calling set.sh for setting back to default"
# c_freq=1380
# ./set.sh $m_freq $c_freq
echo " ========================= END init.sh over!!! ========================= "
echo " "
