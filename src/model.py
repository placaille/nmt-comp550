import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, batch_size,
                 n_layers=2):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(input_size, hidden_size)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)

    def forward(self, input, hidden, input_lengths):
        embedding = self.embedding(input)
        output = pack_padded_sequence(embedding, input_lengths)
        for i in xrange(self.n_layers):
            output, hidden = self.rnn(output, hidden)
        output, output_lengths = pad_packed_sequence(output)
        return output, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        n, b, e = self.n_layers, self.batch_size, self.hidden_size
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

        self.embedding = nn.Embedding(output_size, hidden_size)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)

        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)  # done in masked_cross_ent

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, input.size(0), -1)
        for i in xrange(self.n_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)
        output = self.out(output.view(input.size(0), -1))
        return output, hidden

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

        self.attention = nn.Linear(hidden_size*2, hidden_size)
        self.vector    = nn.Parameter(torch.FloatTensor(hidden_size, 1))

    def forward(self, hidden, hs_encoder):
        length, bs = hs_encoder.size(0), hs_encoder.size(1)
        attn = Variable(torch.zeros(bs, length))

        if hs_encoder.is_cuda :
            attn = attn.cuda()

        # we consider one "word" from the decoder at a time
        if len(hidden.size()) == 3 : 
            assert hidden.size(0) == 1 
            hidden = hidden[0]

        # we iterate over the encoder's states
        for i in xrange(length):
            attn_score = torch.cat((hidden, hs_encoder[i]), dim=-1) # bs x 2dim
            attn_score = self.attention(attn_score) # bs x dim
            attn_score = torch.mm(attn_score, self.vector) # bs x 1
            attn[:, i] = attn_score.squeeze()

        return F.softmax(attn, dim=1).unsqueeze(1) # bs x 1 x length 


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

        output, hidden = self.rnn(embedded, hidden)

        attn_weights = self.attn(output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        output = output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        out = self.out(concat_output)

        return out, hidden, attn_weights
        
        
    def initHidden(self):
        raise NotImplementedError('function not implemented, as we only need \
                to pass in the encoder\'s hidden state to init the decoder')
        
