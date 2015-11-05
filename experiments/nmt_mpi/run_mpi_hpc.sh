#!/bin/bash
#PBS -l nodes=8:ppn=8
#PBS -l walltime=2:00:00
#PBS -q isi

cd /home/nlg-05/xingshi/workspace/misc/lstm/GroundHog/experiments/nmt_mpi

THEANO_FLAGS=floatX=float32 mpiexec -ppn 1 python asyn_mpi.py --proto=prototype_en_zh 1>./hpc_stdout.txt 2>./hpc_stderr.txt
#mode=FAST_COMPILE
