import experiments.nmt_mpi
from experiments.nmt_mpi import\
        RNNEncoderDecoder, prototype_state, get_batch_iterator

state = getattr(experiments.nmt_mpi, 'prototype_en_zh')()
state['sort_k_batches'] = 1

train_data = get_batch_iterator(state)

train_data.start(-1)



