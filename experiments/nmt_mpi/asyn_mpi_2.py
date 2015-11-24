
from mpi4py import MPI
import sys

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

# assume each machine have only 2 gpu
if rank > 2 and rank % 3 == 0:
    sys.exit(0)

# to use gpu
import os
flags = os.environ['THEANO_FLAGS']
if rank != 0:
    flags += ',device=gpu{}'.format(str(rank%3-1))
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
    
    #### get all the workders COMPILE ####
    broadcast_tag(None,tags.COMPILE)
    
    vals = None
    for i in xrange(1,size):
        data = comm.recv(source=MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
        source = status.Get_source()
        tag = status.Get_tag()
        assert(tag == tags.DONE)
        if source == 1:
            vals = data
    
    logger.info('All workers are COMPILED!')
    
    broadcast_tag(None,tags.BEFORE_TRAIN)
    synchronize_tag(tags.DONE)

    logger.info('All workers BEFORE_TRAIN DONE!')

    source_words = 0
    target_words = 0

    start_time = time.time()
    
    nWokers = size - 1
    closed_worders = 0
    
    # first round #
    broadcast_tag(vals,tags.TRAIN_BATCHES)
    stop = False
    while closed_worders < nWokers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.DONE:
            delta,tsw,ttw = data
            source_words += tsw
            target_words += ttw
            till_now = time.time() - start_time
            logger.info('AverageSpeed {}/{}'.format(source_words/till_now,target_words/till_now))
        
            if delta == None:
                if source == 1:
                    logger.info('Worker1 finish its training!')
                else:
                    logger.info('Something went wrong on workers !')
                stop = True
            if stop:
                comm.send(None,dest=source,tag=tags.EXIT)
            else:
                vals = train_mpi.add_delta(vals,delta)
                logger.info('Sending jobs to {} worker'.format(source))
                comm.send(vals,dest=source,tag=tags.TRAIN_BATCHES)
        if tag == tags.EXIT:
            closed_worders += 1

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
            if rank == 1:
                context = train_mpi.compile(name+":"+str(rank),isMaster = True)
                vals = train_mpi.get_parameter(context)
                comm.send(vals, dest=0, tag=tags.DONE)
            else:
                context = train_mpi.compile(name+":"+str(rank))
                comm.send(None, dest=0, tag=tags.DONE)
            
        if tag == tags.BEFORE_TRAIN:
            mainloop = context['mainloop']
            if rank == 1:
                train_context = mainloop.before_train_master()
            else:
                train_context = mainloop.before_train_worker(rank-1)
            comm.send(None, dest=0, tag=tags.DONE)

        if tag == tags.TRAIN_BATCHES:
            vals = data
            round_start_time = time.time()
            train_mpi.set_parameter(context,vals)
            mainloop = context['mainloop']
            if rank == 1:
                # interrunpt in child process doesn't work
                stop,interrupt,tsw,ttw = mainloop.train_batches_master(train_context)
                round_time = time.time() - round_start_time
                logger.info('RoundSpeed {}/{}, RoundTime {} sec'.format(tsw/round_time, ttw/round_time,round_time))
        
                stop = stop or interrupt
                if stop:
                    # save the model
                    mainloop.after_train_master(train_context)
                    comm.send((None,tsw,ttw), dest=0, tag=tags.DONE)
                else:
                    new_vals = train_mpi.get_parameter(context)
                    delta = train_mpi.get_delta(new_vals,vals)
                    comm.send((delta,tsw,ttw), dest=0, tag=tags.DONE)

            else:

                success,tsw,ttw = mainloop.train_batches_worker(train_context)
                round_time = time.time() - round_start_time
                logger.info('RoundSpeed {}/{}, RoundTime {} sec'.format(tsw/round_time, ttw/round_time,round_time))

                new_vals = train_mpi.get_parameter(context)
                if not success:
                    comm.send((None,tsw,ttw), dest=0, tag=tags.DONE)
                else:
                    delta = train_mpi.get_delta(new_vals,vals)
                    comm.send((delta,tsw,ttw), dest=0, tag=tags.DONE)

        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)





if rank == 0:
    master()
else:
    worker()
