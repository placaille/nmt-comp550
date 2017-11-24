import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, rnn_type, input_size, embd_size, hidden_size, batch_size, n_layers=2):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embd_size  = embd_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(input_size, embd_size)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(embd_size, hidden_size, n_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embd_size, hidden_size, n_layers)

    def forward(self, input, hidden):
        output = self.embedding(input).view(input.size(0), input.size(1), -1)
        for i in xrange(self.n_layers):
            output, hidden = self.rnn(output, hidden)
        return output, hidden

    def init_hidden_state(self):
        hidden_state_dims = (self.n_layers, self.batch_size, self.hidden_size)
        if self.rnn_type == 'LSTM':
            return (Variable(torch.zeros(hidden_state_dims)),
                    Variable(torch.zeros(hidden_state_dims)))
        else:
            return Variable(torch.zeros(hidden_state_dims))


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)


        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
