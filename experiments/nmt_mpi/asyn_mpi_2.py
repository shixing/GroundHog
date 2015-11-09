from mpi4py import MPI

def enum(*sequential, **named):
    """
    Handy way to fake an enumerated type in Python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'COMPILE', 'BEFORE_TRAIN','TRAIN_BATCHES', 'DONE', 'EXIT')


# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

# to use gpu
import os
flags = os.environ['THEANO_FLAGS']
flags += ',device=gpu{}'.format(str(rank%2))
flags += ',base_compiledir=/home/nlg-05/xingshi/.theano/{}'.format(str(rank))
os.environ['THEANO_FLAGS'] = flags

print flags

#import theano.sandbox.cuda
#if rank == 0:
#    theano.sandbox.cuda.use('gpu'+str(rank%2))

import logging
import time
import train_mpi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format="%(asctime)s: "+str(rank)+" %(name)s: %(levelname)s: %(message)s")


if rank == 0:
    #pass
    #data = comm.recv(source=MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
    train_mpi.compile("0",isMaster=True)
    #import theano
else:
    #import theano
    #import theano
    train_mpi.compile(str(rank))

