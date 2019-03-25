#!/bin/bash
#Name of job
#Dep name , project name
#PBS -N cross_val
#PBS -P cse

### To use High Priority issue the following commands with PBS command
    #-q high
    #-P mausam.inlp.cs

#PBS -j oe
#PBS -l select=1:ngpus=1:K20GPU=false:ncpus=4
#PBS -l walltime=24:00:00
## SPECIFY JOB NOW

JOBNAME=cross_val
CURTIME=$(date +%Y%m%d%H%M%S)
cd $PBS_O_WORKDIR
## Change to dir from where script was launched

