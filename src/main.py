# coding: utf-8
from __future__ import division
import os
import pdb
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import utils  # custom file with lors of functions used
import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='../data/multi30k',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    choices=['LSTM', 'GRU'],
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--max_length', type=int, default=50, metavar='N',
                    help='maximal sentence length')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--use-attention', action='store_true',
                    help='use attention mechanism in the decoder')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--lang', type=str,  default='en-fr',
                    choices=['en-fr'],
                    help='in-out languages')
parser.add_argument('--verbose', action='store_true',
                    help='verbose flag')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

if args.verbose:
    print('Processing data..')
corpus = data.Corpus(args.data, args.lang)

###############################################################################
# Build the model
###############################################################################

if args.verbose:
    print('Building model..')
encoder = model.EncoderRNN(args.model,
                           len(corpus.dictionary['src']),
                           args.nhid,
                           args.batch_size,
                           args.nlayers)

if args.use_attention:
    decoder = model.AttentionDecoderRNN(
                           args.model, 
                           args.nhid, 
                           len(corpus.dictionary['tgt']),
                           args.batch_size, 
                           args.nlayers)
else:
    decoder = model.DecoderRNN(
                           args.model,
                           args.nhid,
                           len(corpus.dictionary['tgt']),
                           args.batch_size,
                           args.nlayers)

if args.cuda:
    encoder.cuda()
    decoder.cuda()

if args.verbose:
    print(encoder)
    print(decoder)

criterion = nn.NLLLoss()
lr = args.lr
enc_optim = torch.optim.Adam(encoder.parameters(), lr)
dec_optim = torch.optim.Adam(decoder.parameters(), lr)

###############################################################################
# Training code
###############################################################################


def train_epoch():
    # Turn on training mode which enables dropout.
    encoder.train()
    decoder.train()

    utils.set_gradient(encoder, True)
    utils.set_gradient(decoder, True)

    total_loss = 0
    start_time = time.time()

    # initialize minibatch generator
    minibatches = utils.minibatch_generator(args.batch_size,
                                            corpus.train,
                                            args.cuda)

    for n_batch, batch in enumerate(minibatches):

        loss, _ = utils.step(encoder, decoder, batch, enc_optim, dec_optim,
                             criterion, args.use_attention, True, args.cuda,
                             args.max_length, args.clip)

        total_loss += loss

        if n_batch % args.log_interval == 0 and n_batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                  epoch, n_batch, corpus.n_sent_train // args.batch_size, lr,
                  elapsed * 1000 / args.log_interval, cur_loss, np.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if args.verbose:
    print('Starting training..')

# Loop over epochs.
best_val_loss = None
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_epoch()
        val_loss = utils.evaluate(corpus.valid, encoder, decoder,
                                  args.batch_size, args.use_attention,
                                  args.cuda, criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, np.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(args.save, 'encoder.pt'), 'wb') as f:
                torch.save(encoder, f)
            with open(os.path.join(args.save, 'decoder.pt'), 'wb') as f:
                torch.save(decoder, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(os.path.join(args.save, 'encoder.pt'), 'rb') as f:
    encoder = torch.load(f)
with open(os.path.join(args.save, 'decoder.pt'), 'rb') as f:
    decoder = torch.load(f)

# Run on test data.
test_loss = utils.evaluate(corpus.test, encoder, decoder, args.batch_size,
                           args.use_attention, args.cuda, criterion)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, np.exp(test_loss)))
print('=' * 89)
