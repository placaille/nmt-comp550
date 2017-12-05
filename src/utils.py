# coding: utf-8
from __future__ import division
import numpy as np
import pdb
import random
import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.autograd import Variable
from torch.nn import functional

from beam_wrapper import *


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = Variable(torch.LongTensor(length)).cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    try:
        log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    except:
        # It seems that by default log_softmax sums to 1 along last dimension
        log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)

    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def step(encoder, decoder, batch, optimizer,
         train=True, cuda=True, max_length=50, clip=0, tf_p=0.,
         beam_size=0):

    PAD_token = 0
    SOS_token = 2

    batch_src, batch_tgt, len_src, len_tgt = batch

    max_src = batch_src.size(0)
    max_tgt = batch_tgt.size(0)
    try:
        b_size = batch_tgt.size(1)
    except:
        b_size = 1
        batch_src.unsqueeze(1)
        batch_tgt.unsqueeze(1)

    loss = 0
    use_teacher_forcing = False

    if train:
        optimizer.zero_grad()
        use_teacher_forcing = True if np.random.random() < tf_p else False

    enc_h0 = encoder.init_hidden(b_size)

    # run src sentence in encoder and get final state
    enc_out, context = encoder(batch_src, enc_h0, len_src)

    # reshape hidden state if bidirectional
    if encoder.bidirectional:
        if encoder.rnn_type == 'LSTM':
            n2, b, h = context[0].size()
            dec_hid = tuple([context[0].view(n2 // 2, b, h * 2).contiguous(),
                             context[1].view(n2 // 2, b, h * 2).contiguous()])
        else:
            n2, b, h = context.size()
            dec_hid = context.view(n2 // 2, b, h * 2).contiguous()
    else:
        dec_hid = context

    # create SOS tokens for decoder input
    dec_input = Variable(torch.LongTensor([SOS_token] * b_size))

    # Create variable that will hold all the sequence from decoding
    dec_outs = Variable(torch.zeros(max_tgt, b_size, decoder.output_size))
    preds = torch.LongTensor(max_tgt, b_size).zero_()

    decoder_attentions = torch.zeros(b_size, max_src, max_tgt)

    if cuda:
        dec_input = dec_input.cuda()
        dec_outs = dec_outs.cuda()
        preds = preds.cuda()

    # decode by looping time steps
    if beam_size:

        beam_searcher = BSWrapper(decoder, dec_hid, b_size, max_length,
                                  beam_size, cuda, enc_out)

        preds = torch.LongTensor(beam_searcher.decode())

        return 0, preds

    else:

        for step in xrange(max_tgt):

            dec_out, dec_hid, attn_weights = decoder(dec_input, dec_hid, enc_out)

            if decoder.use_attention:

                decoder_attentions[:, :attn_weights.size(2), step] += attn_weights.squeeze().cpu().data

                # decoder_attentions[:, :attn_weights.size(1), step] += attn_weights.squeeze().cpu().data

            # get highest scoring token and value
            top_val, top_tok = dec_out.data.topk(1, dim=1)
            if use_teacher_forcing:
                dec_input = batch_tgt[step].unsqueeze(-1)
            else:
                dec_input = Variable(top_tok)

            # store all steps for later loss computing
            dec_outs[step] = dec_out
            preds[step] = top_tok

        loss = masked_cross_entropy(dec_outs.transpose(1, 0).contiguous(),
                                    batch_tgt.transpose(1, 0).contiguous(),
                                    len_tgt)

        # update params
        if train:
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm(encoder.parameters(), clip)
                nn.utils.clip_grad_norm(decoder.parameters(), clip)
            optimizer.step()

    return loss.data[0], preds, decoder_attentions


def set_gradient(model, value):
    for p in model.parameters():
        p.requires_grad = value


def minibatch_generator(size, dataset, cuda, shuffle=True):
    """
    Generator used to feed the minibatches
    """
    PAD_token = 0

    def fill_seq(input, padded_length, fill_token):
        input_padded = input[:]
        input_padded += [fill_token] * (padded_length - len(input))
        return input_padded

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

        '''
        Simple test to see if attention mechanism is working
        ind = torch.arange(batch_src.size(0) - 1, -1, -1).long()
        ind = ind.cuda() if cuda else ind
        batch_tgt = batch_src[ind]
        len_tgt_s = len_src_s[::-1]
        # import pdb; pdb.set_trace()
        yield batch_src, batch_src, len_src_s, len_src_s
        '''
        yield batch_src, batch_tgt, len_src_s, len_tgt_s

def evaluate(dataset, encoder, decoder, args, corpus=None):
    # Turn on evaluation mode which disables dropout.
    encoder.eval()
    decoder.eval()

    set_gradient(encoder, False)
    set_gradient(decoder, False)

    total_loss = 0
    iters = 0

    # initialize minibatch generator
    minibatches = minibatch_generator(args.batch_size, dataset, args.cuda)

    upper_bd = 10 if args.debug else float('inf')

    for n_batch, batch in enumerate(minibatches):

        loss, dec_outs, attn = step(encoder, decoder, batch, optimizer=None,
                                    train=False, cuda=args.cuda,
                                    max_length=args.max_length)

        total_loss += loss
        iters += 1

        if n_batch > upper_bd: break

    if args.show_attention and args.use_attention: 
        for i in range(2):
            batch_src, batch_tgt, len_src, len_tgt = batch
            src, tgt = batch_src[:, i], batch_tgt[:, i]
            src_sentence = [corpus.dictionary['src'].idx2word[x] for x in src.data]
            tgt_sentence = [corpus.dictionary['tgt'].idx2word[x] for x in tgt.data]
            att_sentence = attn[i].transpose(1,0)
            show_attention(src_sentence, tgt_sentence, att_sentence, name=i)

    loss = total_loss / iters
    return loss, dec_outs


def show_attention(input_sentence, output_words, attentions, name=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')

    if type(input_sentence) == type(''):
        input_sentence = input_sentence.split(' ')

    # set up axes
    ax.set_xticklabels([''] + input_sentence + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if name is not None : 
        if not os.path.exists('images'):
            os.makedirs('images')
        plt.savefig('images/' + str(name) + '.png')
    else : 
        plt.show()
    plt.close()
