import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable


def build_model(src_vocab_size, tgt_vocab_size, args):

    encoder = EncoderRNN(args.model,
                         src_vocab_size,
                         args.nhid,
                         args.batch_size,
                         args.nlayers,
                         args.bidirectional)

    if args.use_attention:
        decoder = AttentionDecoderRNN(args.model,
                                      args.nhid,
                                      tgt_vocab_size,
                                      args.batch_size,
                                      args.max_length,
                                      args.nlayers,
                                      args.dropout)
    else:
        decoder = DecoderRNN(args.model,
                             args.nhid,
                             tgt_vocab_size,
                             args.batch_size,
                             args.nlayers)

    return encoder, decoder


class EncoderRNN(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, batch_size,
                 n_layers=2, bidirectional=False):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, hidden_size)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, 
                    bidirectional=bidirectional)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, 
                    bidirectional=bidirectional)

    def forward(self, input, hidden, input_lengths):
        embedding = self.embedding(input)
        output = pack_padded_sequence(embedding, input_lengths)
        for i in xrange(self.n_layers):
            output, hidden = self.rnn(output, hidden)
        output, output_lengths = pad_packed_sequence(output)

        # sum of multiple directions if applicable  directions
        if self.bidirectional:
            output = output[:, :, self.hidden_size:] + \
                     output[:, :, :self.hidden_size]
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        n, b, e = self.n_layers, batch_size, self.hidden_size
        if self.bidirectional:
            n *= 2
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(n, b, e).zero_()),
                    Variable(weight.new(n, b, e).zero_()))
        else:
            return Variable(weight.new(n, b, e).zero_())


class DecoderRNN(nn.Module):
    def __init__(self, rnn_type, hidden_size, output_size, batch_size, n_layers=2):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.use_attention = False

        self.embedding = nn.Embedding(output_size, hidden_size)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)

        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)  # done in masked_cross_ent

    def forward(self, input, hidden, encoder_outputs=None):
        output = self.embedding(input).view(1, input.size(0), -1)
        for i in xrange(self.n_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)
        output = self.out(output.view(input.size(0), -1))
        return output, hidden, None

    def init_hidden(self):
        # TODO fix changing batch size at end of epoch
        weight = next(self.parameters()).data
        n, b, e = self.n_layers, self.batch_size, self.hidden_size
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(n, b, e).zero_()),
                    Variable(weight.new(n, b, e).zero_()))
        else:
            return Variable(weight.new(n, b, e).zero_())


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(Attention, self).__init__()
        self.dense = nn.Linear(hidden_size*2, hidden_size)


    def forward(self, hidden_state, encoder_outputs):

        # make sure inputs have the same batch size
        assert hidden_state.size(1) == encoder_outputs.size(1)
    
        assert len(hidden_state.size()) == 3 


        '''
        build a batch x len_target x len_source tensor
        note that len_targer should == 1, as were calculating
        the attention for 1 "word" at a time
        '''
        grid = torch.bmm(hidden_state.transpose(1,0), 
                         encoder_outputs.permute(1,2,0)) 

        '''
        to have valid weights / probs, we need that our tensor sums 
        to 1 over the encoder outpus (dim=1). We need to perform
        a masked softmax in order to discard the padding
        '''
        mask = (grid != 0).float().cuda()
        attn_weights = F.softmax(grid, dim=2) * mask
        normalizer = attn_weights.sum(dim=2).unsqueeze(2)
        attn_weights /= normalizer


        '''
        once we have the attention weights, apply them to your 
        context in order to extract the relevant features from it. 
        This is where the conditional extraction takes place
        '''

        weighted_context = torch.bmm(attn_weights, 
                                     encoder_outputs.transpose(1,0))

        '''
        we merge our (weighted) context with the original input
        '''

        concat = torch.cat((weighted_context, hidden_state.transpose(1,0)), -1)

        out = F.tanh(self.dense(concat))

        # b x 1 x dim --> 1 x b x dim
        return out.transpose(1,0), attn_weights


class AttentionDecoderRNN(nn.Module):
    def __init__(self, rnn_type, hidden_size, output_size, batch_size, 
                 max_length=50, n_layers=2, dropout_p=0.1):
        super(AttentionDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size  = batch_size
        self.n_layers    = n_layers
        self.rnn_type    = rnn_type
        self.dropout_p   = dropout_p
        self.use_attention = True

        self.embedding    = nn.Embedding(output_size, hidden_size)
        self.attn         = Attention(hidden_size, batch_size)
        self.dropout      = nn.Dropout(dropout_p)
        self.out          = nn.Linear(hidden_size, output_size)
        self.concat       = nn.Linear(hidden_size*2, hidden_size)

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)


    def forward(self, input, hidden, encoder_outputs):

        embedded = self.embedding(input).view(1, input.size(0), -1)
        embedded = self.dropout(embedded)

        # use this as input for yout rnn
        attn_weights, softmax_over_input = self.attn(embedded, encoder_outputs)
 
        output = F.relu(attn_weights)
        output, hidden = self.rnn(output, hidden)
        out = self.out(output).squeeze(0)

        return out, hidden, softmax_over_input



