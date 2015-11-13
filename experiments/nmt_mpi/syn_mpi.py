
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
name = MPI.Get_processor_name()

# to use gpu
import os
flags = os.environ['THEANO_FLAGS']
flags += ',device=gpu{}'.format(str(rank%2))
flags += ',base_compiledir=/home/nlg-05/xingshi/.theano/{}/{}'.format(name,str(rank))
#flags += ',base_compiledir=/tmp/.theano/{}'.format(str(rank))
os.environ['THEANO_FLAGS'] = flags

print flags

import logging
import time
import train_mpi

logger = logging.getLogger(__name__)


def synchronize_tag(tag):
    for i in xrange(1,size):
        data = comm.recv(source=MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
        tag = status.Get_tag()
        assert(tag == tag)
    
def broadcast_tag(data,tag):
    for i in xrange(1,size):
        comm.send(data,dest=i,tag=tag)

def master():

    name = MPI.Get_processor_name()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s: "+name+":0"+" %(name)s: %(levelname)s: %(message)s")
    logger.info("I am master with rank %d on %s." % (rank,name))


    # get all the workers READY signal
    synchronize_tag(tags.READY)
    logger.info("All workers are READY!")
    
    #### COMPILE ####
    
    broadcast_tag(None,tags.COMPILE)
    
    context = train_mpi.compile(name+":0",isMaster=True)
    mainloop = context['mainloop']
    
    synchronize_tag(tags.DONE)    
    logger.info('All workers COMPILE done!')

    #### Train #####

    # before_train
    
    broadcast_tag(None,tags.BEFORE_TRAIN)
    train_context = mainloop.before_train_master()
    synchronize_tag(tags.DONE)

    source_words = 0
    target_words = 0
    start_time = time.time()
    vals = None
    while True:
        # broadcast values and send to each child
        round_start_time = time.time()

        round_source_words = 0
        round_target_words = 0

        if vals == None:
            vals = train_mpi.get_parameter(context)

        bstart = time.time()
        broadcast_tag(vals,tags.TRAIN_BATCHES)
        bend = time.time()
        logger.info('Broadcast Parameters to {} nodes cost {} seconds'.format(size-1,bend-bstart))

        stop,interrupt,tsw,ttw = mainloop.train_batches_master(train_context)
        
        round_source_words += tsw
        round_target_words += ttw

        if interrupt:
            mainloop.after_train_master(train_context)
            return

        vals = train_mpi.get_parameter(context)

        for i in xrange(1,size):
            worker_vals,worker_tsw,worker_ttw = comm.recv(source=MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
            tag = status.Get_tag()
            assert(tag == tags.DONE)
            assert(worker_vals != None)

            round_source_words += worker_tsw
            round_target_words += worker_ttw

            for name in worker_vals:
                master_val = vals[name]
                worker_val = worker_vals[name]
                master_val += worker_val        

        for name in vals:
            vals[name] /= size
        
        train_mpi.set_parameter(context,vals)
        
        source_words += round_source_words
        target_words += round_target_words

        round_end_time = time.time()
        round_time = round_end_time - round_start_time
        till_now = round_end_time - start_time

        logger.info('RoundSpeed {}/{}, AverageSpeed {}/{}, RoundTime {} sec'.format(round_source_words/round_time, round_target_words/round_time,source_words/till_now,target_words/till_now,round_time))
        
        if stop:
            break

    #### Finish Training ####
    mainloop.after_train_master(train_context)

    for i in xrange(1,size):
        comm.send(None, dest = i , tag = tags.EXIT)
        
    synchronize_tag(tags.EXIT)

    logger.info('All workers EXIT!')

    logger.info('Master EXIT!')
    

def worker():
    name = MPI.Get_processor_name()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s: "+name+":"+str(rank)+" %(name)s: %(levelname)s: %(message)s")
    logger.info("I am a worker with rank %d on %s." % (rank, name))
    comm.send(None, dest=0, tag=tags.READY)
    
    context = None
    train_context = None

    while True:        
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        
        if tag == tags.COMPILE:
            context = train_mpi.compile(name+":"+str(rank))
            comm.send(None, dest=0, tag=tags.DONE)
            
        if tag == tags.BEFORE_TRAIN:
            mainloop = context['mainloop']
            train_context = mainloop.before_train_worker(rank)
            comm.send(None, dest=0, tag=tags.DONE)

        if tag == tags.TRAIN_BATCHES:
            vals = data
            train_mpi.set_parameter(context,vals)
            mainloop = context['mainloop']
            success,tsw,ttw = mainloop.train_batches_worker(train_context)
            vals = train_mpi.get_parameter(context)
            if not success:
                vals = None
            comm.send((vals,tsw,ttw), dest=0, tag=tags.DONE)

        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)





if rank == 0:
    master()
else:
    worker()
