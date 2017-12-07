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
                         args.emb_size,
                         args.batch_size,
                         args.nlayers,
                         args.bidirectional)

    if encoder.bidirectional:
        dec_nhid = args.nhid * 2
    else:
        dec_nhid = args.nhid

    if args.use_attention:
        decoder = Luong_Decoder(args.model,
                                dec_nhid,
                                tgt_vocab_size,
                                args.batch_size,
                                n_layers=args.nlayers,
                                dropout_p=args.dropout)

    else:
        decoder = DecoderRNN(args.model,
                             dec_nhid,
                             args.emb_size,
                             tgt_vocab_size,
                             args.batch_size,
                             args.nlayers)

    return encoder, decoder


class EncoderRNN(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, embedding_size, batch_size,
                 n_layers=2, bidirectional=False):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, embedding_size)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, n_layers,
                              bidirectional=bidirectional)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers,
                               bidirectional=bidirectional)

    def forward(self, input, hidden, input_lengths):
        embedding = self.embedding(input)
        embedding_packed = pack_padded_sequence(embedding, input_lengths)

        output, hidden = self.rnn(embedding_packed, hidden)
        output, output_lengths = pad_packed_sequence(output)

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
    def __init__(self, rnn_type, hidden_size, embedding_size, output_size,
                 batch_size, n_layers=2):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.use_attention = False

        self.embedding = nn.Embedding(output_size, embedding_size)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, n_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs=None):
        embedding = self.embedding(input).view(1, input.size(0), -1)

        output, hidden = self.rnn(embedding, hidden)
        output = self.out(output.view(input.size(0), -1))
        return output, hidden, None

class Luong_Attention(nn.Module):
    def __init__(self, hidden_size, score='general'):
        super(Luong_Attention, self).__init__()

        assert score.lower() in ['concat', 'general', 'dot']
        self.score = score.lower()
        wn = lambda x : nn.utils.weight_norm(x)
        
        if self.score == 'general': 
            self.attn = wn(nn.Linear(hidden_size, hidden_size))
        elif self.score == 'concat':
            raise Exception('concat disabled for now. results are poor')
            self.attn = wn(nn.Linear(2 * hidden_size, hidden_size))
            self.v = wn(nn.Linear(hidden_size, 1))


    def forward(self, hidden_state, encoder_outputs):

        # make sure inputs have the same batch size
        assert hidden_state.size(1) == encoder_outputs.size(1)

        assert len(hidden_state.size()) == 3 

        # put batch on 1st axis (easier for batch matrix mul)
        hidden_state    = hidden_state.transpose(1, 0).contiguous()
        encoder_outputs = encoder_outputs.transpose(1, 0).contiguous()

        if self.score == 'dot': 
            # bs x tgt_len=1 x src_len
            grid = torch.bmm(hidden_state, encoder_outputs.transpose(2,1)) 
        elif self.score == 'general': 
            # bs x tgt_len=1 x src_len
            grid = torch.bmm(hidden_state, self.attn(encoder_outputs).transpose(2,1))
        elif self.score == 'concat':
            # bs x src_len x n_hid
            cc = self.attn(torch.cat((hidden_state.expand(encoder_outputs.size()), 
                                      encoder_outputs), 2))
            # bs x src_len x 1
            grid = self.v(cc)
            # bs x tgt_len=1 x n_hid
            grid = grid.permute(0, 2, 1)

        # make sure to compute softmax over valid tokens only
        mask = (grid != 0).float()
        attn_weights = F.softmax(grid, dim=2) * mask
        normalizer = attn_weights.sum(dim=2).unsqueeze(2)
        attn_weights /= normalizer

        return attn_weights


class Luong_Decoder(nn.Module):
    def __init__(self, rnn_type, hidden_size, output_size, batch_size, 
                 max_length=50, n_layers=2, dropout_p=0.1):
        super(Luong_Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size  = batch_size
        self.n_layers    = n_layers
        self.rnn_type    = rnn_type
        self.dropout_p   = dropout_p
        self.use_attention = True
        wn = lambda x : nn.utils.weight_norm(x)

        self.embedding    = nn.Embedding(output_size, hidden_size)
        self.attn         = Luong_Attention(hidden_size)
        self.dropout      = nn.Dropout(dropout_p)
        self.out          = wn(nn.Linear(hidden_size, output_size))
        self.concat       = wn(nn.Linear(hidden_size*2, hidden_size))

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, input.size(0), -1)
        embedded = self.dropout(embedded)
        # use this as input for yout rnn
        
        rnn_output, hidden = self.rnn(embedded, hidden)
        attn_weights  = self.attn(rnn_output, encoder_outputs)
        # attn_weights : bs x 1 x src_len
        # enc_outputs  : src_len x bs x nhid
        context = encoder_outputs * attn_weights.permute(2, 0, 1)
        # bs x nhid
        context = context.sum(dim=0)
        concat_input  = torch.cat((context, rnn_output.squeeze(0)), 1)
        concat_output = F.tanh((self.concat(concat_input)))

        out = self.out(concat_output)

        return out, hidden, attn_weights
