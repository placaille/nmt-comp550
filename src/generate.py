# coding: utf-8
from __future__ import division
import os
import pdb
import argparse
import time
import torch
import pickle as pkl

import utils  # custom file with lors of functions used
import data
import model

parser = argparse.ArgumentParser()
parser.add_argument('--data_src', type=str,
                    default='../out/multi30k/en-fr/val.en')
parser.add_argument('--data_tgt', type=str,
                    default='../out/multi30k/en-fr/val.fr')
parser.add_argument('--path_to_model', type=str,
                    default='../out/temp_run/bin',
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--max_length', type=int, default=50, metavar='N',
                    help='maximal sentence length')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--beam_size', type=int, default=5, metavar='N',
                    help='width of beam search during generation')
parser.add_argument('--path_to_word2vec', type=str,
                    default='./GoogleNews-vectors-negative300.bin.gz',
                    help='Path to pre-trained word2vec')
parser.add_argument('--lang', type=str,  default='en-fr',
                    choices=['en-fr', 'en-de'],
                    help='in-out languages')
parser.add_argument('--verbose', action='store_true',
                    help='verbose flag')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load training run args
with open(os.path.join(args.path_to_model, '../args.info'), 'rb') as f:
    train_args = pkl.load(f)

###############################################################################
# Load data
###############################################################################

if args.verbose:
    print('Processing data..')

# load the dictionary for generation
with open(os.path.join(args.path_to_model, 'vocab.pt'), 'rb') as f:
    dictionary = pkl.load(f)
corpus = data.GenerationCorpus(dictionary, args.data_src, args.data_tgt,
                               train_args.reverse_src)

###############################################################################
# Load the model
###############################################################################

if args.verbose:
    print('Loading model from {}..'.format(args.path_to_model))

assert os.path.exists(os.path.join(args.path_to_model, 'encoder_params.pt'))
assert os.path.exists(os.path.join(args.path_to_model, 'decoder_params.pt'))

# create new model of same specs as training
encoder, decoder = model.build_model(len(corpus.dictionary['src']),
                                     len(corpus.dictionary['tgt']),
                                     args=train_args)

# Load the best saved model.
with open(os.path.join(args.path_to_model, 'encoder_params.pt'), 'rb') as f:
    encoder.load_state_dict(torch.load(f))
with open(os.path.join(args.path_to_model, 'decoder_params.pt'), 'rb') as f:
    decoder.load_state_dict(torch.load(f))

if args.cuda:
    encoder.cuda()
    decoder.cuda()

if args.verbose:
    print(encoder)
    print(decoder)

###############################################################################
# Predicting code
###############################################################################


def make_preds(dataset, encoder, decoder, dictionary, batch_size,
               cuda, max_length):

    # Turn on evaluation mode which disables dropout.
    encoder.eval()
    decoder.eval()

    utils.set_gradient(encoder, False)
    utils.set_gradient(decoder, False)

    pred_corpus_tokens = []
    gold_corpus_tokens = []
    start_time = time.time()

    minibatches = utils.minibatch_generator(batch_size, dataset, cuda)
    for n_batch, batch in enumerate(minibatches):

        _, pred, _ = utils.step(encoder, decoder, batch, None, train=False,
                                cuda=cuda, max_length=max_length,
                                beam_size=args.beam_size)

        # true target
        gold = batch[1].data

        for i in xrange(pred.size(1)):
            # get tokens from the predicted iindices
            pred_tokens = [dictionary.idx2word[x] for x in pred[:, i]]
            gold_tokens = [dictionary.idx2word[x] for x in gold[:, i]]

            # filter out u'<pad>', u'<eos>'
            filter_tokens = [u'<pad>', u'<eos>']
            pred_tokens = filter(lambda x: x not in filter_tokens, pred_tokens)
            gold_tokens = filter(lambda x: x not in filter_tokens, gold_tokens)

            pred_corpus_tokens.append(pred_tokens)
            gold_corpus_tokens.append(gold_tokens)

        if n_batch % args.log_interval == 0 and n_batch > 0:
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | ms/batch {:5.2f} |'.format(
                  n_batch, len(dataset[0]) // args.batch_size, 
                  elapsed * 1000 / args.log_interval))
            start_time = time.time()

    return pred_corpus_tokens, gold_corpus_tokens


if args.verbose:
    print('Making predictions..')

pred_dir = os.path.join(args.path_to_model, '../pred')
gold_dir = os.path.join(args.path_to_model, '../gold')

pred_name = os.path.basename(args.data_src).split('.')[0]
gold_name = os.path.basename(args.data_tgt).split('.')[0]

pred_file = os.path.join(pred_dir, 'pred_{}_{}.txt'.format(pred_name, args.lang))
gold_file = os.path.join(gold_dir, 'gold_{}_{}.txt'.format(gold_name, args.lang))
pred_file_nounk = os.path.join(pred_dir, 'pred_{}_{}_nounk.txt'.format(pred_name, args.lang))
gold_file_nounk = os.path.join(gold_dir, 'gold_{}_{}_nounk.txt'.format(gold_name, args.lang))

pred_tokens, gold_tokens = make_preds(corpus.gen_dataset, encoder, decoder,
                                      corpus.dictionary['tgt'],
                                      args.batch_size, args.cuda,
                                      args.max_length)

with open(pred_file, 'w') as f, open(gold_file, 'w') as g:
    for pred_tok, gold_tok in zip(pred_tokens, gold_tokens):
        f.write(' '.join(pred_tok).encode('utf8') + '\n')
        g.write(' '.join(gold_tok).encode('utf8') + '\n')

print('{} was saved.'.format(pred_file))
print('{} was saved.'.format(gold_file))

with open(pred_file_nounk, 'w') as f, open(gold_file_nounk, 'w') as g:
    for pred_tok, gold_tok in zip(pred_tokens, gold_tokens):
        # filter out sentences that had "unknown" token in gold
        if u'<unk>' not in gold_tok:
            f.write(' '.join(pred_tok).encode('utf8') + '\n')
            g.write(' '.join(gold_tok).encode('utf8') + '\n')

print('{} was saved.'.format(pred_file_nounk))
print('{} was saved.'.format(gold_file_nounk))

print('=' * 89)
