#!/usr/bin/python2.7

import numpy as np
import multiprocessing as mp
import Queue
from utils.dataset import Dataset
from utils.network import Forwarder
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
from utils.viterbi import Viterbi


### helper function for parallelized Viterbi decoding ##########################
def decode(queue, log_probs, decoder, index2label):
    while not queue.empty():
        try:
            video = queue.get(timeout = 3)
            score, labels, segments = decoder.decode( log_probs[video] )
            # save result
            with open('results/' + video, 'w') as f:
                f.write( '### Recognized sequence: ###\n' )
                f.write( ' '.join( [index2label[s.label] for s in segments] ) + '\n' )
                f.write( '### Score: ###\n' + str(score) + '\n')
                f.write( '### Frame level recognition: ###\n')
                f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )
        except Queue.Empty:
            pass


### read label2index mapping and index2label mapping ###########################
label2index = dict()
index2label = dict()
with open('data/mapping.txt', 'r') as f:
    content = f.read().split('\n')[0:-1]
    for line in content:
        label2index[line.split()[1]] = int(line.split()[0])
        index2label[int(line.split()[0])] = line.split()[1]

### read test data #############################################################
with open('data/split1.test', 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = Dataset('data', video_list, label2index, shuffle = False)

# load prior, length model, grammar, and network
load_iteration = 10000
log_prior = np.log( np.loadtxt('results/prior.iter-' + str(load_iteration) + '.txt') )
grammar = PathGrammar('results/grammar.txt', label2index)
length_model = PoissonModel('results/lengths.iter-' + str(load_iteration) + '.txt', max_length = 2000)
forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
forwarder.load_model('results/network.iter-' + str(load_iteration) + '.net')

# parallelization
n_threads = 8

# Viterbi decoder
viterbi_decoder = Viterbi(grammar, length_model, frame_sampling = 30)
# forward each video
log_probs = dict()
queue = mp.Queue()
for i, data in enumerate(dataset):
    sequence, _ = data
    video = dataset.features.keys()[i]
    queue.put(video)
    log_probs[video] = forwarder.forward(sequence) - log_prior
    log_probs[video] = log_probs[video] - np.max(log_probs[video])
# Viterbi decoding
procs = []
for i in range(n_threads):
    p = mp.Process(target = decode, args = (queue, log_probs, viterbi_decoder, index2label) )
    procs.append(p)
    p.start()
for p in procs:
    p.join()

