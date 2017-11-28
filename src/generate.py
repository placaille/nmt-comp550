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
parser.add_argument('--lang', type=str,  default='en-fr',
                    choices=['en-fr'],
                    help='in-out languages')
parser.add_argument('--verbose', action='store_true',
                    help='verbose flag')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###############################################################################
# Load data
###############################################################################

if args.verbose:
    print('Processing data..')

# load the dictionary for generation
with open(os.path.join(args.path_to_model, 'vocab.pt'), 'rb') as f:
    dictionary = pkl.load(f)
corpus = data.GenerationCorpus(dictionary, args.data_src, args.data_tgt)

###############################################################################
# Load the model
###############################################################################

if args.verbose:
    print('Loading model from {}..'.format(args.path_to_model))

# Load the best saved model.
assert os.path.exists(os.path.join(args.path_to_model, 'encoder.pt'))

with open(os.path.join(args.path_to_model, 'encoder.pt'), 'rb') as f:
    encoder = torch.load(f)
with open(os.path.join(args.path_to_model, 'decoder.pt'), 'rb') as f:
    decoder = torch.load(f)

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

    pred_corpus = []
    gold_corpus = []
    start_time = time.time()

    minibatches = utils.minibatch_generator(batch_size, dataset, cuda)
    for n_batch, batch in enumerate(minibatches):

        _, pred = utils.step(encoder, decoder, batch, None, None, train=False,
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
            pred_corpus.append(' '.join(filter(
                lambda x: x not in filter_tokens, pred_tokens)))
            gold_corpus.append(' '.join(filter(
                lambda x: x not in filter_tokens, gold_tokens)))

        if n_batch % args.log_interval == 0 and n_batch > 0:
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | ms/batch {:5.2f} |'.format(
                  n_batch, len(dataset[0]) // args.batch_size, 
                  elapsed * 1000 / args.log_interval))
            start_time = time.time()

    return pred_corpus, gold_corpus


if args.verbose:
    print('Making predictions..')

pred_dir = os.path.join(args.path_to_model, '../pred')
gold_dir = os.path.join(args.path_to_model, '../gold')

pred_file = os.path.join(pred_dir, 'pred_{}.txt'.format(os.path.basename(args.data_tgt)))
gold_file = os.path.join(gold_dir, 'gold_{}.txt'.format(os.path.basename(args.data_src)))

pred_corpus, gold_corpus = make_preds(corpus.gen_dataset, encoder, decoder,
                                      corpus.dictionary['tgt'],
                                      args.batch_size, args.cuda,
                                      args.max_length)

with open(pred_file, 'w') as f:
    for sentence in pred_corpus:
        f.write(sentence.encode('utf8') + '\n')

with open(gold_file, 'w') as f:
    for sentence in gold_corpus:
        f.write(sentence.encode('utf8') + '\n')

if args.verbose:
    print('{} was saved.'.format(pred_file))
    print('{} was saved.'.format(gold_file))
    print('=' * 89)

