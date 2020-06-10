#!/bin/bash
for i in {12..23}; do 
    CUDA_VISIBLE_DEVICES=1 bash exp_$i.sh ; 
done

