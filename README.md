# nmt-comp550

This repo is built on top and highly inspired by pytorch's
word\_language\_model example, and their seq2seq tutorial. See [this
repo](https://github.com/pytorch/examples/tree/master/word_language_model) and
[the
doc](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
for reference.

The code for the masked_cross_entropy was borrowed from [this
repo](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation).


## Links to download the data

For more information refer to [this link](http://www.statmt.org/wmt17/multimodal-task.html).

The gold predictions for the task can be [found
here](http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/gold_translations_task1.tar.gz).
However, these are already normalized, tokenized and lower cased. Furthermore,
for French, the apostrophe character was replaced by the string "\&apos; ". As
they are our only version of the French 2017 test, we manually modified these
strings to make them like true characters that a French model should predict.

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


## Pre-trained word embeddings

Multiple pre-trained word embeddings model can be used. Here are a couple
options.

* Google's word2vec model, more information can be
found [at this link](https://code.google.com/archive/p/word2vec/). Direct
download of the matrix for the 3 million words can be done with this [direct
link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).

* Stanford's GloVe model, more information can
be found [at this link](https://nlp.stanford.edu/projects/glove/). Direct
download of pre-trained model (42B tokens, 1.9M vocab, uncased, 300d vectors)
can be done with this [direct
link](http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip).

To use it in python, we will be using the `gensim` package. Refer to [this
function](https://github.com/placaille/nmt-comp550/blob/master/src/utils.py#L19) in our code for detailed usage of the pre-trained model.

### References

One of the first if not the first seq2seq learning, they talk about the
necessity of doing beam search. See *Decoding and Rescoring* [link to
paper](https://arxiv.org/pdf/1409.3215v1.pdf%3B)

Beam search implementation from [this repo](
https://github.com/GuessWhatGame/guesswhat/blob/master/src/guesswhat/models/qgen/qgen_beamsearch_wrapper.py) was highly inspired but was modified to fit our needs and model.

