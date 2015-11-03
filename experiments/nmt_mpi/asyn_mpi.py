from mpi4py import MPI

import train_mpi

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

    print("I am master with rank %d on %s." % (rank,name))

    # get all the workers READY signal
    synchronize_tag(tags.READY)
    print('All workers are READY!')
    
    #### COMPILE ####

    broadcast_tag(None,tags.COMPILE)

    context = train_mpi.compile(isMaster=True)
    mainloop = context['mainloop']

    synchronize_tag(tags.DONE)
    print('All workers COMPILE done!')
    

    #### Train #####

    # before_train
    
    broadcast_tag(None,tags.BEFORE_TRAIN)
    train_context = mainloop.before_train_master()
    synchronize_tag(tags.DONE)

    vals = None
    while True:
        # broadcast values and send to each child
        if vals == None:
            vals = train_mpi.get_parameter(context)

        broadcast_tag(vals,tags.TRAIN_BATCHES)
        stop,interrupt = mainloop.train_batches_master(train_context)

        
        if stop:
            break

        if interrupt:
            mainloop.after_train_master(train_context)
            return

        vals = train_mpi.get_parameter(context)

        for i in xrange(1,size):
            worker_vals = comm.recv(source=MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
            tag = status.Get_tag()
            assert(tag == tags.DONE)
            assert(worker_vals != None)
            for name in worker_vals:
                master_val = vals[name]
                worker_val = worker_vals[name]
                master_val += worker_val

        print('All workers TRAIN_BATCHES done!')

        for name in vals:
            vals[name] /= size
        
        train_mpi.set_parameter(context,vals)
        


    #### Finish Training ####
    mainloop.after_train_master(train_context)

    for i in xrange(1,size):
        comm.send(None, dest = i , tag = tags.EXIT)
        
    synchronize_tag(tags.EXIT)
    print('All workers EXIT!')

    print('Master EXIT!')
    

def worker():

    print("I am a worker with rank %d on %s." % (rank, name))
    comm.send(None, dest=0, tag=tags.READY)
    
    context = None
    train_context = None

    while True:        
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        
        if tag == tags.COMPILE:
            context = train_mpi.compile()
            comm.send(None, dest=0, tag=tags.DONE)
            
        if tag == tags.BEFORE_TRAIN:
            mainloop = context['mainloop']
            train_context = mainloop.before_train_worker(rank)
            comm.send(None, dest=0, tag=tags.DONE)

        if tag == tags.TRAIN_BATCHES:
            vals = data
            train_mpi.set_parameter(context,vals)
            mainloop = context['mainloop']
            success = mainloop.train_batches_worker(train_context)
            vals = train_mpi.get_parameter(context)
            if not success:
                vals = None
            comm.send(vals, dest=0, tag=tags.DONE)

        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)



if rank == 0:
    master()
else:
    worker()
