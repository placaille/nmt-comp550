# -*- coding: utf-8 -*-
import os
import re
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.vocab_set = set()

        # add <unk> <sos> and <eos> tokens
        self.add_word(u'<eos>')  # ID 0
        self.add_word(u'<sos>')  # ID 1
        self.add_word(u'<unk>')  # ID 2

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, lang):

        path = os.path.join(path, lang)

        self.dictionary = {'src': Dictionary(), 'tgt': Dictionary()}
        lang_src, lang_tgt = lang.split('-')

        train_src = self.tokenize(os.path.join(path,
            'train.{}'.format(lang_src)), 'src')
        train_tgt = self.tokenize(os.path.join(path,
            'train.{}'.format(lang_tgt)), 'tgt')
        valid_src = self.tokenize(os.path.join(path,
            'val.{}'.format(lang_src)), 'src')
        valid_tgt = self.tokenize(os.path.join(path,
            'val.{}'.format(lang_tgt)), 'tgt')
        test_src = self.tokenize(os.path.join(path,
            'test.{}'.format(lang_src)), 'src')
        test_tgt = self.tokenize(os.path.join(path,
            'test.{}'.format(lang_tgt)), 'tgt')

        self.train = (train_src, train_tgt)
        self.valid = (valid_src, valid_tgt)
        self.test = (test_src, test_tgt)

    def tokenize(self, path, src_tgt):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            sentences = 0
            max_tokens = 0
            for line in f:
                line = line.decode('utf-8', 'strict')
                words = re.findall(r"[\w']+|[.,!?;]", line.lower(),
                        flags=re.UNICODE) + [u'<eos>']

                # only add words if in training set
                if 'train' in path:
                    for word in words:
                        self.dictionary[src_tgt].add_word(word)
                    self.dictionary[src_tgt].vocab_set = \
                        set(self.dictionary[src_tgt].idx2word)

                # track stats for building tokenized version
                tokens = len(words)
                sentences += 1
                if tokens > max_tokens:
                    max_tokens = tokens

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(max_tokens, sentences).zero_()
            for i, line in enumerate(f):
                token = 0
                line = line.decode('utf-8', 'strict')
                words = re.findall(r"[\w']+|[.,!?;]", line.lower(),
                        flags=re.UNICODE) + [u'<eos>']
                for word in words:
                    if word not in self.dictionary[src_tgt].vocab_set:
                        word = u'<unk>'
                    ids[token, i] = self.dictionary[src_tgt].word2idx[word]
                    token += 1

        return ids
