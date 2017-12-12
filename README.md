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

**Text data**
* [Train](http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_training.tar.gz)
* [Valid](http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_validation.tar.gz)
* [Test2016](http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz)
* [Test2017](http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/source_flickr.task1)

**Average pooled features from ResNet-50**
* [Train](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/flickr30k_ResNet50_pool5_train.zip)
* [Valid](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/flickr30k_ResNet50_pool5_val.zip)
* [Test2016](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/flickr30k_ResNet50_pool5_test.zip)
* [Test2017](http://www-lium.univ-lemans.fr/sites/default/files/NMTPY/test2017/task1_ResNet50_pool5_test2017.mat.zip)


#### Processing of data
Data is lowecased and tokenized, punctuation are considered as tokens. Only
training data is used to build the vocabulary (from both source and target languages), while words not in vocabulary are replaced by the \<unk\> tag.


### References

One of the first if not the first seq2seq learning, they talk about the
necessity of doing beam search. See *Decoding and Rescoring* [link to
paper](https://arxiv.org/pdf/1409.3215v1.pdf%3B)

Other paper about beam search strategies [link to
paper](https://arxiv.org/pdf/1702.01806.pdf)

Beam search implementation from [this repo](
https://github.com/GuessWhatGame/guesswhat/blob/master/src/guesswhat/models/qgen/qgen_beamsearch_wrapper.py)

Highly inspired but was modified to fit our needs and model

