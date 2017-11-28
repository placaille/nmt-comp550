# coding: utf-8
from __future__ import division
import numpy as np
import pdb
import random
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional

from Beam import *  # OpenNMT-py Beam implementation


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


def step(encoder, decoder, batch, enc_optim, dec_optim,
         train=True, cuda=True, max_length=50, clip=0, tf_p=0.,
         beam_size=5):

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
        enc_optim.zero_grad()
        dec_optim.zero_grad()
        use_teacher_forcing = True if np.random.random() < tf_p else False

    enc_h0 = encoder.init_hidden(b_size)

    # run src sentence in encoder and get final state
    enc_out, enc_hid = encoder(batch_src, enc_h0, len_src)
    dec_hid = enc_hid
    dec_input = Variable(torch.LongTensor([SOS_token] * b_size))

    # Create variable that will hold all the sequence from decoding
    dec_outs = Variable(torch.zeros(max_length, b_size, decoder.output_size))
    preds = torch.LongTensor(max_length, b_size).zero_()

    if cuda:
        dec_input = dec_input.cuda()
        dec_outs = dec_outs.cuda()
        preds = preds.cuda()

    if beam_size:

        # The following is loosely base on OpenNMT-py Translator's
        # implementation. it was modified to reflect our implementation

        k = beam_size

        # copy all encoding and initial inputs to decoder for beam size

        if len(dec_hid) == 2:
            # it's a lstm
            dec_hid_ = tuple([dec_hid[0].repeat(1, k, 1),
                             dec_hid[1].repeat(1, k, 1)])
        else:
            dec_hid_ = dec_hid.repeat(1, k, 1)

        dec_outs_ = dec_outs.repeat(1, k, 1)
        preds_ = preds.repeat(1, k, 1)

        if decoder.use_attention:
            enc_out_ = enc_out.repeat(1, k, 1)

        beam = [Beam(k, 1, cuda) for _ in range(b_size)]

        for step in xrange(max_length):

            # check if <eos> for all beams
            if all((b.done() for b in beam)):
                break

            # stack the batch and beams to compute in one take
            dec_input_ = Variable(torch.stack(
                [b.getCurrentState() for b in beam]).view(-1))

            if decoder.use_attention:
                dec_out_, dec_hid_, _ = decoder(dec_input_, dec_hid_,
                                                enc_out_)
            else:
                dec_out_, dec_hid_ = decoder(dec_input_, dec_hid_)

            # compute log_probs for each batch and beam
            log_p = functional.log_softmax(dec_out_.view(k, b_size, -1), dim=2)

            pdb.set_trace()
            # avance each beam
            for j, b in enumerate(beam):
                b.advance(log_p[:, j])

                dec_states





        # create tensors used for selecting best tokens
        scores = torch.zeros(k ** 2)
        step_scores = torch.zeros(max_length, k ** 2)
        step_tokens = torch.zeros(max_length, k)

        if cuda:
            scores = scores.cuda()
            step_scores = step_scores.cuda()

        if decoder.use_attention:
            enc_out_ = enc_out[:, i].unsqueeze(1).repeat(1, k)

        for step in xrange(max_length):
            if decoder.use_attention:
                dec_out_, dec_hid_, _ = decoder(dec_input_, dec_hid_,
                        enc_out_)
            else:
                dec_out_, dec_hid_ = decoder(dec_input_, dec_hid_)

            # we need to pick the top k, keep the info
            new_log_probs = functional.log_softmax(dec_out_, dim=1)
            top_val, top_tok = new_log_probs.data.topk(k, dim=1)

            step_scores[step] = top_val.view(-1)

            scores += step_scores[step]

            # now that we have the scores of all new full sentences
            # get the k best and we use these as the input to next step
            best_total_scores, best_idx_tok = scores.topk(k)

            beam_token = torch.gather(top_tok.view(-1), 0, best_idx_tok)
            step_tokens[step] = beam_token
            dec_input_ = Variable(beam_token)

            pdb.set_trace()

        return 0, preds

    else:
        # no beam search, do greedy decode by looping time steps
        for step in xrange(max_length):
            if decoder.use_attention:
                dec_out, dec_hid, _ = decoder(dec_input, dec_hid, enc_out)
            else:
                dec_out, dec_hid = decoder(dec_input, dec_hid)

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
            enc_optim.step()
            dec_optim.step()

    return loss.data[0], preds


def set_gradient(model, value):
    for p in model.parameters():
        p.requires_grad = value


def minibatch_generator(size, dataset, cuda, shuffle=True):
    """
    Generator used to feed the minibatches
    """
    PAD_token = 0

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


def evaluate(dataset, encoder, decoder, batch_size, cuda, max_length):
    # Turn on evaluation mode which disables dropout.
    encoder.eval()
    decoder.eval()

    set_gradient(encoder, False)
    set_gradient(decoder, False)

    total_loss = 0
    iters = 0

    # initialize minibatch generator
    minibatches = minibatch_generator(batch_size, dataset, cuda)
    for n_batch, batch in enumerate(minibatches):

        loss, dec_outs = step(encoder, decoder, batch, None, None,
                              train=False, cuda=cuda,
                              max_length=max_length)

        total_loss += loss
        iters += 1

    loss = total_loss / iters
    return loss, dec_outs
