#!/bin/bash

free_mem_0=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0| grep -Eo [0-9]+)
free_mem_1=$(nvidia-smi --query-gpu=memory.free --format=csv -i 1| grep -Eo [0-9]+)

gpu_id=2
while true 
do
    free_mem_0=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0| grep -Eo [0-9]+)
    free_mem_1=$(nvidia-smi --query-gpu=memory.free --format=csv -i 1| grep -Eo [0-9]+)
        if [ "$free_mem_0" -gt "22000" ]; then
            gpu_id=0
            break
        fi
        if [ "$free_mem_1" -gt "22000" ]; then
            gpu_id=1
            break
        fi
    sleep 200
done  
CUDA_VISIBLE_DEVICES=$gpu_id python $1
 
