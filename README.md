# nmt-comp550

This repo is built on top and highly inspired by pytorch's
word\_language\_model example. See [this
repo](https://github.com/pytorch/examples/tree/master/word_language_model)
for reference.

## The task is the WMT17 multimodal (task 1)
For more information refer to [this
link](http://www.statmt.org/wmt17/multimodal-task.html)


For simplicity, French characters with accents were normalized to the letter
only.

#### Processing of data
Data is lowecased and tokenized, punctuation are considered as tokens. Only
training data is used to build the vocabulary (from both source and target languages), while words not in vocabulary are replaced by the \<unk\> tag.

Processed data is a tensor of the indices to word dictionary, arranged in the
form [tokens, sentences] of size max\_tokens x nb\_sentences, where we pad the shorter sentences with \<eos\> tag (indice 0). we then have tuples for each train, valid and test datasets.


### Model input
The input to the encoder is a matrix of the tokens ID for each sequence
(columns is the batch axis). The embedding layer of pytorch maps the ID to an
embedding of the same dimensions as the hidden state.

