# -*- coding: utf-8 -*-
import os
import unicodedata
import re
import torch
import pdb

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

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

        self.train_in = self.tokenize(os.path.join(path,
            'train.{}'.format(lang_in)))
        self.train_out = self.tokenize(os.path.join(path,
            'train.{}'.format(lang_out)))
        self.valid_in = self.tokenize(os.path.join(path,
            'val.{}'.format(lang_in)))
        self.valid_out = self.tokenize(os.path.join(path,
            'val.{}'.format(lang_out)))
        self.test_in = self.tokenize(os.path.join(path,
            'test.{}'.format(lang_in)))
        self.test_out = self.tokenize(os.path.join(path,
            'test.{}'.format(lang_out)))

    def sub_french_accents(string):
        """
        Thanks to stackoverflow for the idea
        """
        string = re.sub(u"[èéêë]", 'e', string)
        string = re.sub(u"[òóôõö]", 'o', string)
        string = re.sub(u"[ìíîï]", 'i', string)
        string = re.sub(u"[ùúûü]", 'u', string)
        string = re.sub(u"[àáâãäå]", 'a', string)
        string = re.sub(u"[ýÿ]", 'y', string)
        return string

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            pdb.set_trace()
            tokens = 0
            for line in f:
                line = line.decode('utf-8', 'strict')
                words = re.findall(r"[\w']+|[.,!?;]", line.lower(),
                        flags=re.UNICODE) + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line = line.decode('utf-8', 'strict')
                words = re.findall(r"[\w']+|[.,!?;]", line.lower(),
                        flags=re.UNICODE) + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
