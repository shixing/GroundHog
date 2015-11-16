import argparse
import cPickle
import logging
import pprint

import numpy

from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.trainer.SGD import SGD as SGD
from groundhog.trainer.SGD_momentum import SGD as SGD_momentum
from groundhog.mainLoop import MainLoop
from groundhog.mainLoop_mpi import MainLoop_mpi
from experiments.nmt_mpi import\
        RNNEncoderDecoder, prototype_state, get_batch_iterator
import experiments.nmt_mpi

logger = logging.getLogger(__name__)
    
class RandomSamplePrinter(object):

    def __init__(self, state, model, train_iter):
        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

    def __call__(self):
        def cut_eol(words):
            for i, word in enumerate(words):
                if words[i] == '<eol>':
                    return words[:i + 1]
            raise Exception("No end-of-line found")

        sample_idx = 0
        while sample_idx < self.state['n_examples']:
            batch = self.train_iter.next(peek=True)
            xs, ys = batch['x'], batch['y']
            for seq_idx in range(xs.shape[1]):
                if sample_idx == self.state['n_examples']:
                    break

                x, y = xs[:, seq_idx], ys[:, seq_idx]
                x_words = cut_eol(map(lambda w_idx : self.model.word_indxs_src[w_idx], x))
                y_words = cut_eol(map(lambda w_idx : self.model.word_indxs[w_idx], y))
                if len(x_words) == 0:
                    continue

                print "Input: {}".format(" ".join(x_words))
                print "Target: {}".format(" ".join(y_words))
                self.model.get_samples(self.state['seqlen'] + 1, self.state['n_samples'], x[:len(x_words)])
                sample_idx += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    return parser.parse_args()

def compile(ip,isMaster=False):
    # compile the model

    # context maintains all the necessary variables needed during the training.
    context = {}

    args = parse_args()

    state = getattr(experiments.nmt_mpi, args.proto)()
    if args.state:
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    logging.root.setLevel(getattr(logging, state['level']))
    logger.debug("State:\n{}".format(pprint.pformat(state)))

    rng = numpy.random.RandomState(state['seed'])

    skip_init = args.skip_init

    if not isMaster:
        skip_init = True
    enc_dec = RNNEncoderDecoder(state, rng, skip_init)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    if isMaster:
        if 'master_model_path' in state:
            model_path = state['master_model_path']
            lm_model.load(model_path)
            

    validate_data = None
    if isMaster:
        logger.debug("Load validate data")
        validate_data = get_batch_iterator(state,validation=True)
        logger.debug("build valid fn")
        valid_fn = enc_dec.create_validate_fn(lm_model)
        lm_model.validate_step = valid_fn
        
    logger.debug("Load data")
    train_data = get_batch_iterator(state)
    logger.debug("Compile trainer")
    algo = eval(state['algo'])(lm_model, state, train_data)

    main = MainLoop_mpi(train_data, validate_data, None, lm_model, algo, state, None,
            reset=state['reset'],
            hooks=[RandomSamplePrinter(state, lm_model, train_data)]
                if state['hookFreq'] >= 0 and isMaster
                else None)

    context['state'] = state
    context['train_data'] = train_data
    context['model'] = lm_model
    context['algo'] = algo
    context['isMaster'] = isMaster
    context['mainloop'] = main

    return context

def set_parameter(context,vals):
    model = context['model']
    for p in model.params:
        if p.name in vals:
            logger.debug('Loading {} of {}'.format(p.name, p.get_value(borrow=True).shape))
            if p.get_value().shape != vals[p.name].shape:
                raise Exception("Shape mismatch: {} != {} for {}"
                                .format(p.get_value().shape, vals[p.name].shape, p.name))
            p.set_value(vals[p.name])
        else:
            # FIXME: do not stop loading even if there's a parameter value missing
            #raise Exception("No parameter {} given".format(p.name))
            logger.error( "No parameter {} given: default initialization used".format(p.name))
        unknown = set(vals.keys()) - {p.name for p in model.params}
        if len(unknown):
            logger.error("Unknown parameters {} given".format(unknown))

def get_parameter(context):
    model = context['model']
    vals = dict([(x.name, x.get_value()) for x in model.params])
    return vals

def get_delta(new_vals,old_vals):
    deltas = {}
    for name in new_vals:
        new_val = new_vals[name]
        old_val = old_vals[name]
        delta = new_val - old_val
        deltas[name] = delta
        
    return deltas

def add_delta(vals,deltas):
    for name in vals:
        vals[name] += deltas[name]
    return vals
