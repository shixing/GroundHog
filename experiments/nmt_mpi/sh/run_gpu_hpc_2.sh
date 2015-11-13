#!/bin/bash
#PBS -l nodes=1:gpus=1
#PBS -l walltime=4:00:00
#PBS -q isi

source $NLGHOME/sh/cuda_source.sh

cd /home/nlg-05/xingshi/workspace/misc/lstm/GroundHog/experiments/nmt_mpi

# Eng_Chr
#THEANO_FLAGS="floatX=float32,nvcc.fastmath=True" mpiexec -ppn 2 python asyn_mpi.py --proto=prototype_en_zh 1>./hpc_stdout.txt 2>./hpc_stderr.txt

# Eng_Fre
THEANO_FLAGS="floatX=float32,nvcc.fastmath=True" mpiexec -ppn 1 python asyn_mpi.py --proto=prototype_en_fr_1 1>./hpc1_stdout.txt 2>./hpc1_stderr.txt



