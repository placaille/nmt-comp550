# -*- coding: utf-8 -*-
import os
import re
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.vocab_set = set()

        # add <unk> and <eos> tokens
        self.add_word(u'<eos>')  # ID 0
        self.add_word(u'<unk>')  # ID 1

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

        self.dictionary = Dictionary()
        lang_in, lang_out = lang.split('-')

        train_in = self.tokenize(os.path.join(path,
            'train.{}'.format(lang_in)))
        train_out = self.tokenize(os.path.join(path,
            'train.{}'.format(lang_out)))
        valid_in = self.tokenize(os.path.join(path,
            'val.{}'.format(lang_in)))
        valid_out = self.tokenize(os.path.join(path,
            'val.{}'.format(lang_out)))
        test_in = self.tokenize(os.path.join(path,
            'test.{}'.format(lang_in)))
        test_out = self.tokenize(os.path.join(path,
            'test.{}'.format(lang_out)))

        self.train = (train_in, train_out)
        self.valid = (valid_in, valid_out)
        self.test = (test_in, test_out)

    def tokenize(self, path):
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
                        self.dictionary.add_word(word)
                    self.dictionary.vocab_set = set(self.dictionary.idx2word)

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
                    if word not in self.dictionary.vocab_set:
                        word = u'<unk>'
                    ids[token, i] = self.dictionary.word2idx[word]
                    token += 1

        return ids
