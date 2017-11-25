# coding: utf-8
from __future__ import division
import pdb
import argparse
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/multi30k',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    choices=['LSTM', 'GRU'],
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
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
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
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
corpus = data.Corpus(args.data, args.lang, args.cuda)

train_src, train_tgt = corpus.train
valid_src, valid_tgt = corpus.valid
test_src, test_tgt = corpus.test

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

decoder = model.DecoderRNN(args.model,
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
enc_optim = torch.optim.Adam(encoder.parameters(), args.lr)
dec_optim = torch.optim.Adam(decoder.parameters(), args.lr)

###############################################################################
# Training code
###############################################################################

def train_step(encoder, decoder, batch_src, batch_tgt, enc_optim, dec_optim,
               criterion, SOS_token=1, EOS_token=0, cuda=True, max_length=50,
               clip_norm=0):

    n_step = batch_tgt.size(0)

    batch_src.unsqueeze(1)
    batch_tgt.unsqueeze(1)

    dec_input = Variable(torch.LongTensor([SOS_token]))

    if cuda:
        batch_src = batch_src.cuda()
        batch_tgt = batch_tgt.cuda()
        dec_input = dec_input.cuda()

    loss = 0
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    enc_h0 = encoder.init_hidden()

    # run src sentence in encoder and get final state
    enc_out, enc_hid = encoder(batch_src, enc_h0)
    dec_hid = enc_hid

    # decode by looping time steps
    for step in xrange(n_step):
        dec_out, dec_hid = decoder(dec_input, dec_hid)

        # get highest scoring token and value
        top_val, top_tok = dec_out.data.topk(1)

        # compute loss
        loss += criterion(dec_out, batch_tgt[step])

        # test if predicting end of sentence
        if top_tok[0][0] == EOS_token:
            break
        dec_input = Variable(top_tok)

    # update params
    loss.backward()
    if clip_norm:
        nn.utils.clip_grad_norm(encoder.parameters(), clip_norm)
        nn.utils.clip_grad_norm(decoder.parameters(), clip_norm)
    enc_optim.step()
    dec_optim.step()

    return loss.data[0] / n_step


def train_epoch():
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()

    # one example at the time to test
    r = range(len(train_src))
    random.shuffle(r)
    for batch_id in r:

        batch_src = train_src[batch_id]
        batch_tgt = train_tgt[batch_id]

        loss = train_step(encoder, decoder, batch_src, batch_tgt, enc_optim, dec_optim, criterion,
                          SOS_token=1, EOS_token=0, cuda=True, max_length=50,
                          clip_norm=None)

        total_loss += loss

        if batch_id % args.log_interval == 0 and batch_id > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_id, len(train_src) // args.batch_size, args.lr,
                elapsed * 1000 / args.log_interval, cur_loss, np.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


if args.verbose:
    print('Starting training..')

# Loop over epochs.
best_val_loss = None
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_epoch()
        # val_loss = evaluate(val_data)
        val_loss = 0
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, np.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
# test_loss = evaluate(test_data)
test_loss = 0
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, np.exp(test_loss)))
print('=' * 89)
