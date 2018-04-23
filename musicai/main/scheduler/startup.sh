#!/bin/sh
clear
pythonpath=""
pythonpath="$(pwd | awk '{split($0,a,"/"); for (i = 2; i<length(a)-2;i++) printf "/" a[i] ;}')"

cd ../../../
PYTHONPATH=$pythonpath python musicai/main/scheduler/input.py > inputlog.log &
#sleep for 1 second for input to create the shared array
sleep 1
PYTHONPATH=$pythonpath python musicai/main/scheduler/output.py > output.log &
