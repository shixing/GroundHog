#THEANO_FLAGS="floatX=float32,device=gpu" python asyn.py --proto=prototype_en_zh
THEANO_FLAGS="floatX=float32,nvcc.fastmath=True" mpiexec -n 2 python asyn_mpi.py --proto=prototype_en_zh
