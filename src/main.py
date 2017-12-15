# coding: utf-8
from __future__ import division
import os
import pdb
import argparse
import time
import numpy as np
import torch
import pickle as pkl

import utils  # custom file with lors of functions used
import data
import model

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data/multi30k',
                    help='location of the data corpus')
parser.add_argument('--path_word_emb', type=str,
                    default='./word_embeddings',
                    help='Path with pre-trained word_embeddings')
parser.add_argument('--model', type=str, default='LSTM',
                    choices=['LSTM', 'GRU'],
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--lr_patience', type=float, default=0,
                    help='patience between val losses before reducing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--use_word_emb', action='store_true',
                    help='use pre-trained word embeddings')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--bidirectional', action='store_true',
                    help='use bidirectional encoder')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--reverse_src', action='store_true',
                    help='reverse src sequence during encoding')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--teacher_force_prob', type=float, default=0.5,
                    help='probability of teacher forcing')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout where applicable')
parser.add_argument('--show_attention', action='store_true',
                    help='show attention grid after evaluate()')
parser.add_argument('--max_length', type=int, default=50, metavar='N',
                    help='maximal sentence length')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='report interval')
parser.add_argument('--use_attention', action='store_true',
                    help='use attention mechanism in the decoder')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--debug', action='store_true',
                    help='reduce training set size to debug')
parser.add_argument('--lang', type=str,  default='en-fr',
                    choices=['en-fr', 'en-de'],
                    help='in-out languages')
parser.add_argument('--verbose', action='store_true',
                    help='verbose flag')
args = parser.parse_args()

# save args
with open(os.path.join(args.save, '../args.info'), 'wb') as f:
    pkl.dump(args, f)

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

if args.use_word_emb:
    if args.verbose:
        print('Loading Word2Vec model..')

    Word2Vec = utils.init_google_word2vec_model(args.path_word_emb)
else:
    Word2Vec = None

if args.verbose:
    print('Processing data..')
corpus = data.Corpus(args.data, args.lang, word_emb=Word2Vec)

# save the dictionary for generation
with open(os.path.join(args.save, 'vocab.pt'), 'wb') as f:
    pkl.dump(corpus.dictionary, f)

###############################################################################
# Build the model
###############################################################################

if args.verbose:
    print('Building model..')

# build model
encoder, decoder = model.build_model(len(corpus.dictionary['src']),
                                     len(corpus.dictionary['tgt']),
                                     args=args)

if args.cuda:
    encoder.cuda()
    decoder.cuda()

if args.verbose:
    print(encoder)
    print(decoder)


lr = args.lr
optimizer = torch.optim.Adam(list(encoder.parameters()) +
                             list(decoder.parameters()),
                             lr)

# scheduler to reduce the lr by 4 (*0.25) if val loss doesn't decr for 2 epoch
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.25,
                                                       verbose=True,
                                                       patience=args.lr_patience)

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
                                            args.cuda,
                                            word2vec=Word2Vec,
                                            vocab=corpus.dictionary['src'],
                                            shuffle=True)

    upper_bd = 10 if args.debug else float('inf')
    for n_batch, batch in enumerate(minibatches):

        loss, _, _ = utils.step(encoder, decoder, batch, optimizer, True, args)
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

        if n_batch > upper_bd : break


if args.verbose:
    print('Starting training..')

# Loop over epochs.
best_val_loss = float('inf')
best_epoch = 0
# At any point you can hit Ctrl + C to break out of training early.
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train_epoch()
    val_loss, _ = utils.evaluate(corpus.valid, encoder, decoder, args, corpus=corpus)
    scheduler.step(val_loss)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'val ppl {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                   val_loss, np.exp(val_loss)))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_epoch = epoch
        if args.verbose:
            print('| Saving model parameters..')

        enc_path = os.path.join(args.save, 'encoder_params.pt')
        dec_path = os.path.join(args.save, 'decoder_params.pt')
        torch.save(encoder.state_dict(), enc_path)
        torch.save(decoder.state_dict(), dec_path)

        best_val_loss = val_loss
    print('-' * 89)

print('=' * 89)
if args.verbose:
    print('| Loading best model for epoch {} and evaluating test..'
          .format(best_epoch))

# create new model of same specs
best_encoder, best_decoder = model.build_model(len(corpus.dictionary['src']),
                                               len(corpus.dictionary['tgt']),
                                               args=args)

if args.cuda:
    best_encoder.cuda()
    best_decoder.cuda()

# Load the best saved model params
best_enc_path = os.path.join(args.save, 'encoder_params.pt')
best_dec_path = os.path.join(args.save, 'decoder_params.pt')

best_encoder.load_state_dict(torch.load(best_enc_path))
best_decoder.load_state_dict(torch.load(best_dec_path))

# Run on test data.
test2016_loss, _ = utils.evaluate(corpus.test2016, encoder, decoder, args, corpus=corpus)
test2017_loss, _ = utils.evaluate(corpus.test2017, encoder, decoder, args, corpus=corpus)
print('| End of training | test 2016 loss {:5.2f} | test 2016 ppl {:8.2f}'
      .format(test2016_loss, np.exp(test2016_loss)))
print('| End of training | test 2017 loss {:5.2f} | test 2017 ppl {:8.2f}'
      .format(test2017_loss, np.exp(test2017_loss)))
print('=' * 89)
