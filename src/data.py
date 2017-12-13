# -*- coding: utf-8 -*-
import os
import pdb
import h5py
import numpy as np
import re
import torch
import pickle as pkl
from torch.autograd import Variable

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.vocab_set = set()

        # add <unk> <sos> and <eos> tokens
        self.add_word(u'<pad>')  # ID 0
        self.add_word(u'<eos>')  # ID 1
        self.add_word(u'<sos>')  # ID 2
        self.add_word(u'<unk>')  # ID 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, lang, reverse_src=False, load_img_feat=False):

        path = os.path.join(path, lang)

        self.reverse_src = reverse_src
        self.dictionary = {'src': Dictionary(), 'tgt': Dictionary()}
        lang_src, lang_tgt = lang.split('-')

        train_src = self.tokenize(os.path.join(path,
            'train.{}'.format(lang_src)), 'src', True)
        train_tgt = self.tokenize(os.path.join(path,
            'train.{}'.format(lang_tgt)), 'tgt', True)
        valid_src = self.tokenize(os.path.join(path,
            'val.{}'.format(lang_src)), 'src', False)
        valid_tgt = self.tokenize(os.path.join(path,
            'val.{}'.format(lang_tgt)), 'tgt', False)
        test2016_src = self.tokenize(os.path.join(path,
            'test2016.{}'.format(lang_src)), 'src', False)
        test2016_tgt = self.tokenize(os.path.join(path,
            'test2016.{}'.format(lang_tgt)), 'tgt', False)
        test2017_src = self.tokenize(os.path.join(path,
            'test2017.{}'.format(lang_src)), 'src', False)
        test2017_tgt = self.tokenize(os.path.join(path,
            'test2017.{}'.format(lang_tgt)), 'tgt', False)
        
        if load_img_feat:
            path_to_feat = os.path.join(path, '../image_features')
            path_train = os.path.join(path_to_feat, 'flickr30k_ResNet50_pool5_train.mat')
            path_val = os.path.join(path_to_feat, 'flickr30k_ResNet50_pool5_val.mat')
            path_test_16 = os.path.join(path_to_feat, 'flickr30k_ResNet50_pool5_test.mat')
            path_test_17 = os.path.join(path_to_feat, 'task1_ResNet50_pool5_test2017.mat')
            train_img_feat = self.load_img_features(path_train)
            
            val_img_feat = self.load_img_features(path_val)
            test_16_img_feat = self.load_img_features(path_test_16)
            test_17_img_feat = self.load_img_features(path_test_17)

            self.train = (train_src, train_tgt, train_img_feat)
            self.valid = (valid_src, valid_tgt, val_img_feat)
            self.test2016 = (test2016_src, test2016_tgt, test_16_img_feat)
            self.test2017 = (test2017_src, test2017_tgt, test_17_img_feat)
        
        else : 
            self.train = (train_src, train_tgt)
            self.valid = (valid_src, valid_tgt)
            self.test2016 = (test2016_src, test2016_tgt)
            self.test2017 = (test2017_src, test2017_tgt)

        self.n_sent_train = len(train_src)
        self.n_sent_valid = len(valid_src)
        self.n_sent_test2016 = len(test2016_src)
        self.n_sent_test2017 = len(test2017_src)


    def tokenize(self, path, src_tgt, train=False):
        """Tokenizes a text file."""
        assert os.path.exists(path), '{} does not exist'.format(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            sentences = 0
            max_tokens = 0
            for line in f:
                line = line.decode('utf-8', 'strict')
                words = re.findall(r"[\w']+|[.,!?;]", line.lower(),
                        flags=re.UNICODE) + [u'<eos>']

                # only add words if in training set
                if train:
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
            ids = []
            for i, line in enumerate(f):
                line = line.decode('utf-8', 'strict')
                words = re.findall(r"[\w']+|[.,!?;]", line.lower(),
                        flags=re.UNICODE) + [u'<eos>']

                token = 0
                idx = range(len(words))
                for word in words:
                    if word not in self.dictionary[src_tgt].vocab_set:
                        word = u'<unk>'
                    idx[token] = self.dictionary[src_tgt].word2idx[word]
                    token += 1

                # reverse src tokens if applicable
                if self.reverse_src and src_tgt == 'src':
                    idx.reverse()

                # create list of lists for easier process later on
                ids.append(idx)

        return ids

    def load_img_features(self, path):
        f = h5py.File(path, 'r')
        data = np.array(f.items()[0][1])
        return data



class GenerationCorpus(Corpus):
    def __init__(self, vocab, src_path, tgt_path, reverse_src):

        self.dictionary = vocab
        self.reverse_src = reverse_src

        src = self.tokenize(src_path, 'src', False)
        tgt = self.tokenize(tgt_path, 'tgt', False)

        self.gen_dataset = (src, tgt)
        self.n_sent = len(src)
