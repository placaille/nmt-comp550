import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, input, hidden):
        output = self.embedding(input).view(input.size(0), 1, -1)
        for i in xrange(self.n_layers):
            output, hidden = self.rnn(output, hidden)
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
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in xrange(self.n_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        n, b, e = self.n_layers, self.batch_size, self.hidden_size
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(n, b, e).zero_()),
                    Variable(weight.new(n, b, e).zero_()))
        else:
            return Variable(weight.new(n, b, e).zero_())
