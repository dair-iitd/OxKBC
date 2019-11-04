#!/bin/bash
for i in {0..11}; do 
    CUDA_VISIBLE_DEVICES=0 bash exp_${i}.sh ; 
done

