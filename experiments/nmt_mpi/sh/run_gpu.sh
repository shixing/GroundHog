# Eng_Chr
#THEANO_FLAGS="floatX=float32,nvcc.fastmath=True" mpiexec -n 2 python asyn_mpi.py --proto=prototype_en_zh

# Eng_Fre
THEANO_FLAGS="floatX=float32,nvcc.fastmath=True" mpiexec -n 1 python asyn_mpi.py --proto=prototype_en_fr 1>./hpc1_stdout.txt 2>./hpc1_stderr.txt
