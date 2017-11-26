# coding: utf-8
from __future__ import division
import os
import pdb
import argparse
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from masked_cross_entropy import *

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

def step(encoder, decoder, batch, enc_optim, dec_optim, criterion, 
                train=True, cuda=True, max_length=args.max_length, clip=0):

    PAD_token = 2
    SOS_token = 1
    EOS_token = 0

    batch_src, batch_tgt, len_src, len_tgt =  batch

    max_src = batch_src.size(0)
    max_tgt = batch_tgt.size(0)
    try:
        b_size = batch_tgt.size(1)
    except:
        b_size = 1
        batch_src.unsqueeze(1)
        batch_tgt.unsqueeze(1)

    loss = 0

    if train : 
        enc_optim.zero_grad()
        dec_optim.zero_grad()

    enc_h0 = encoder.init_hidden()

    # run src sentence in encoder and get final state
    enc_out, enc_hid = encoder(batch_src, enc_h0, len_src)
    dec_hid = enc_hid
    dec_input = Variable(torch.LongTensor([SOS_token] * b_size))

    # Create variable that will hold all the sequence from decoding
    dec_outs = Variable(torch.zeros(max_tgt, b_size, decoder.output_size))

    if cuda:
        dec_input = dec_input.cuda()
        dec_outs = dec_outs.cuda()

    # decode by looping time steps
    for step in xrange(max_tgt):
        if args.use_attention : 
            dec_out, dec_hid, _ = decoder(dec_input, dec_hid, enc_out)
        else :
            dec_out, dec_hid = decoder(dec_input, dec_hid)

        # get highest scoring token and value
        top_val, top_tok = dec_out.data.topk(1, dim=1)
        dec_input = Variable(top_tok)

        # store all steps for later loss computing
        dec_outs[step] = dec_out

    loss = masked_cross_entropy(dec_outs.transpose(1,0).contiguous(),
                                batch_tgt.transpose(1,0).contiguous(),
                                len_tgt)

    # update params
    if train : 
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm(encoder.parameters(), clip)
            nn.utils.clip_grad_norm(decoder.parameters(), clip)
        enc_optim.step()
        dec_optim.step()

    return loss.data[0]


def set_gradient(model, value):
    for p in model.parameters():
        p.requires_grad = value


def minibatch_generator(size, dataset, cuda, shuffle=True):
    """
    Generator used to feed the minibatches
    """
    PAD_token = 2
    SOS_token = 1
    EOS_token = 0

    def fill_seq(input, padded_length, fill_token):
        input += [fill_token] * (padded_length - len(input))
        return input

    src, tgt = dataset

    nb_elem = len(src)
    indices = range(nb_elem)
    if shuffle:
        random.shuffle(indices)
    while nb_elem > 0:

        b_src = []
        b_tgt = []
        len_src = []
        len_tgt = []

        count = 0
        while count < size and nb_elem > 0:
            ind = indices.pop()
            count += 1
            nb_elem -= 1

            b_src.append(src[ind])
            b_tgt.append(tgt[ind])
            len_src.append(len(src[ind]))
            len_tgt.append(len(tgt[ind]))

        # we need to fill shorter sentences to make tensor
        max_src = max(len_src)
        max_tgt = max(len_tgt)
        b_src_ = [fill_seq(seq, max_src, PAD_token) for seq in b_src]
        b_tgt_ = [fill_seq(seq, max_tgt, PAD_token) for seq in b_tgt]

        # sort the lists by len_src for pack_padded_sentence later
        b_sorted = [(x,y,ls,lt) for (x,y,ls,lt) in \
                       sorted(zip(b_src_, b_tgt_, len_src, len_tgt),
                              key=lambda v: v[2],  # using len_src
                              reverse=True)]  # descending order
        # unzip to individual lists
        b_src_s, b_tgt_s, len_src_s, len_tgt_s = zip(*b_sorted)

        # create pytorch variable, transpose to have (seq, batch)
        batch_src = Variable(torch.LongTensor(b_src_s).t())
        batch_tgt = Variable(torch.LongTensor(b_tgt_s).t())

        if cuda:
            batch_src = batch_src.cuda()
            batch_tgt = batch_tgt.cuda()

        yield batch_src, batch_tgt, len_src_s, len_tgt_s


def train_epoch():
    # Turn on training mode which enables dropout.
    encoder.train()
    decoder.train()
    
    set_gradient(encoder, True)
    set_gradient(decoder, True)

    total_loss = 0
    start_time = time.time()

    # initialize minibatch generator
    minibatches = minibatch_generator(args.batch_size, corpus.train, args.cuda)
    for n_batch, batch in enumerate(minibatches):

        loss = step(encoder, decoder, batch, enc_optim, dec_optim, criterion, 
                        train=True, cuda=args.cuda, max_length=50, clip=args.clip)

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


def evaluate(dataset):
    # Turn on evaluation mode which disables dropout.
    encoder.eval()
    decoder.eval()

    set_gradient(encoder, False)
    set_gradient(decoder, False)

    total_loss = 0
    iters = 0
    start_time = time.time()

    # initialize minibatch generator
    minibatches = minibatch_generator(args.batch_size, dataset, args.cuda)
    for n_batch, batch in enumerate(minibatches):

        try:
            loss = step(encoder, decoder, batch, None, None, criterion,
                        train=False, cuda=args.cuda, max_length=50)
        except:
            loss = 0

        total_loss += loss
        iters += 1

    loss = total_loss / iters
    return loss


if args.verbose:
    print('Starting training..')

# Loop over epochs.
best_val_loss = None
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_epoch()
        val_loss = evaluate(corpus.valid)
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
test_loss = evaluate(corpus.test)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, np.exp(test_loss)))
print('=' * 89)
