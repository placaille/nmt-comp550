# Natural Language Processing through Neural Machine Translation

This is the repository for the final project submission of Philippe Lacaille
and Lucas Pagé-Caccia for the _Natural Language
Processing_ class (COMP550) of Fall 2017 at McGill University.

A copy of the final report submitted is included in this repo.

## Links to download the data

For more information about the data refer to [this link](http://www.statmt.org/wmt17/multimodal-task.html).

**Text data**
* [Train](http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_training.tar.gz)
* [Valid](http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_validation.tar.gz)
* [Test2016](http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz)
* [Test2017 (english)](http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/source_flickr.task1)

**Average pooled features from ResNet-50**
* [Train](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/flickr30k_ResNet50_pool5_train.zip)
* [Valid](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/flickr30k_ResNet50_pool5_val.zip)
* [Test2016](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/flickr30k_ResNet50_pool5_test.zip)
* [Test2017](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/test2017/task1_ResNet50_pool5_test2017.mat.zip)


#### Processing of data
Data is lowecased and tokenized, punctuation are considered as tokens. Only
training data is used to build the vocabulary (from both source and target languages), while words not in vocabulary are replaced by the \<unk\> tag.

The gold predictions for the task can be [found
here](http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/gold_translations_task1.tar.gz).
However, these are already normalized, tokenized and lower cased. Furthermore,
for French, the apostrophe and quotes characters were replaced by the strings
"\&apos; " and "\&quot; ". As they are our only version of the French 2017
test, we manually modified these strings to make them like true characters that
a French model should predict.


## Pre-trained word embeddings

Multiple pre-trained word embeddings model can be used. We used the following
model.

* Google's word2vec model, more information can be
found [at this link](https://code.google.com/archive/p/word2vec/). Direct
download of the matrix for the 3 million words can be done with this [direct
link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).

To use it in python, we will be using the `gensim` package. Refer to [this
function](https://github.com/placaille/nmt-comp550/blob/master/src/utils.py#L18) in our code for detailed usage of the pre-trained model.

### Creating pre-trained embedding layer

For simplicity and speed, we loaded only once the full word2vec model and
replaced our embedding layer parameters with the pre-trained embeddings. For
tokens from the corpus not in the word2vec dictionary, we kept the randomly
initialized embedding parameters. The pre-trained layer can be found at [this
link](https://github.com/placaille/nmt-comp550/blob/master/bin/pre-trained_emb_layer.bin)
in the repo.

Therefore, unkown tokens have an embedding that is unique to them and doesn't
change. The pre-trained embedding layer can be loaded with the arg
`--use_word_emb` and training can be resumed with `--train_word_emb {full,
none, partial}` though partial training is not yet supported.

## References

This repo is built on top and highly inspired by pytorch's
word\_language\_model example, and their seq2seq tutorial under the BSD 3-Clause License. See [this
repo](https://github.com/pytorch/examples/tree/master/word_language_model) and
[the
doc](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
for reference.

The code for the masked_cross_entropy was borrowed from [this
repo](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation) under the The MIT License (MIT).

Beam search implementation from [this repo](
https://github.com/GuessWhatGame/guesswhat/blob/master/src/guesswhat/models/qgen/qgen_beamsearch_wrapper.py) under the Apache License was highly inspired but was modified to fit our needs and model.
