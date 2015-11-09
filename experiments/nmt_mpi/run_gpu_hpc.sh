#!/bin/bash
#PBS -l nodes=2:gpus=1
#PBS -l walltime=2:00:00
#PBS -q isi

source $NLGHOME/sh/cuda_source.sh

cd /home/nlg-05/xingshi/workspace/misc/lstm/GroundHog/experiments/nmt_mpi

THEANO_FLAGS="floatX=float32" mpiexec -ppn 1 python asyn_mpi.py --proto=prototype_en_zh 1>./hpc_stdout.txt 2>./hpc_stderr.txt
#mode=FAST_COMPILE





