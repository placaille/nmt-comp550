import pdb
import torch
import numpy as np
import collections

from torch.autograd import Variable
from torch.nn import functional

# The Beam token is the key element of question generation
#    - path are the outputed words
#    - word_id are the next word inputs
#    - decoder_state is the state of the decoder after outputed path[-1]
#    - score is the \sum log(prob(w)) (beam search only)
#    - prev_beam chain the previous beam (to store the hidden state for each outputed words)
#
# great thanks from the following repo for the implementation https://github.com/GuessWhatGame/guesswhat/blob/master/src/guesswhat/models/qgen/qgen_beamsearch_wrapper.py

BeamToken = collections.namedtuple('BeamToken', ['path', 'word_id', 'decoder_state',
                                                 'score', 'prev_beam',
                                                 'encoder_out'])
def unloop_beam_serie(cur_beam):

    # Build the full beam sequence by using the chain-list structure
    sequence = [cur_beam]
    while cur_beam.prev_beam is not None:
        cur_beam = cur_beam.prev_beam
        sequence.append(cur_beam)

    return sequence[::-1]  # reverse sequence


def create_initial_beam(decoder_state, i, enc_out, batch_size=1):

    if len(decoder_state) == 2:
        dec_hid = tuple([decoder_state[0][:, i].unsqueeze(1).contiguous(),
                         decoder_state[1][:, i].unsqueeze(1).contiguous()])
    else:
        dec_hid = decoder_state[:, i].unsqueeze(1).contiguous()

    if enc_out is not None:
        # means we are using attention
        enc_out = enc_out[:, i].unsqueeze(1).contiguous()

    return BeamToken(
        path=[[] for _ in range(batch_size)],
        word_id=[[] for _ in range(batch_size)],
        decoder_state=dec_hid,
        encoder_out=enc_out,
        score=0,  # initial probability is 1. If we apply the log trick log(1) = 0
        prev_beam=None
    )


class BSWrapper(object):
    def __init__(self, decoder, decoder_state, batch_size, max_length,
                 beam_size, cuda, enc_out):

        self.decoder = decoder
        self.cuda = cuda
        self.PAD_token = 0
        self.EOS_token = 1
        self.SOS_token = 2

        self.max_length=max_length
        self.k_best = beam_size
        self.max_length = max_length
        self.batch_size = batch_size

        if decoder.use_attention:
            enc_out = enc_out
        else:
            enc_out=None

        self.beam = [create_initial_beam(decoder_state, i, enc_out)
                     for i in range(batch_size)]

    def decode(self):

        init_input = [np.array([self.SOS_token]) for _ in range(self.batch_size)]

        for i, one_beam in enumerate(self.beam):

            # Prepare beam by appending answer and removing previous path
            one_beam.word_id[0].append(init_input[i][0])
            one_beam.path[0] = list()

            # Execute beam search
            new_beam = self.eval_one_beam_search(one_beam)

            # Store current beam (with RNN state)
            self.beam[i] = new_beam

        # Compute output
        tokens =  [b.path[0] for b in self.beam]
        seq_length = [len(q) for q in tokens]

        tokens_pad = np.full((len(self.beam), max(seq_length)),
                fill_value=self.PAD_token)
        for i, (q, l) in enumerate(zip(tokens, seq_length)):
            tokens_pad[i, :l] = q

        return tokens_pad.transpose()


    def eval_one_beam_search(self, initial_beam, keep_trajectory=False):

        to_evaluate = [initial_beam]

        memory = []
        for depth in range(self.max_length):

            # evaluate all the current tokens
            for beam_token in to_evaluate:

                # if token is final token, directly put it into memory
                if beam_token.word_id[0][-1] == self.EOS_token:
                    memory.append(beam_token)
                    continue

                dec_input = Variable(torch.LongTensor(beam_token.word_id))
                dec_hid = beam_token.decoder_state
                enc_out = beam_token.encoder_out

                if self.cuda:
                    dec_input = dec_input.cuda()

                # evaluate next_step
                dec_out, dec_hid, attn_weights = self.decoder(dec_input, dec_hid, enc_out)

                # Reshape tensor (remove 1 size batch)
                log_p = functional.log_softmax(dec_out, 1)
                log_p_np = log_p.data.cpu().numpy()[0]

                # put into memory the k-best tokens of this sample
                k_best_word_indices = np.argpartition(log_p_np, -self.k_best)[-self.k_best:]
                for word_id in k_best_word_indices:
                    memory.append(
                        BeamToken(
                            path=[beam_token.path[0] + [word_id]],
                            word_id=[[word_id]],
                            decoder_state=dec_hid,
                            encoder_out=enc_out,
                            score=beam_token.score + log_p_np[word_id],  # log trick
                            prev_beam=beam_token if keep_trajectory else None  # Keep trace of the previous beam if we want to keep the trajectory
                        ))

            # retrieve best beams in memory
            scores = [beam.score / len(beam.path[0]) for beam in memory]
            k_best_word_indices = np.argpartition(scores, -self.k_best)[-self.k_best:]
            to_evaluate = [memory[i] for i in k_best_word_indices]

            # reset memory
            memory = []

        # Pick the best beam
        final_scores = [beam.score / len(beam.path[0]) for beam in to_evaluate]
        best_beam_index = np.argmax(final_scores)
        best_beam = to_evaluate[best_beam_index]

        return best_beam




