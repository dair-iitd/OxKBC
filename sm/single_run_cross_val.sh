#!/bin/bash
#Name of job
#Dep name , project name
#PBS -N Hyperparameter_search
#PBS -P cse
#PBS -j oe
#PBS -l select=1:ngpus=1:K20GPU=false:ncpus=4
#PBS -l walltime=20:00:00
## SPECIFY JOB NOW

JOBNAME=Hyperparameter_search
CURTIME=$(date +%Y%m%d%H%M%S)
cd $PBS_O_WORKDIR
## Change to dir from where script was launched

