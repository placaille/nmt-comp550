# nmt-comp550

This repo is built on top and highly inspired by pytorch's
word\_language\_model example, and their seq2seq tutorial. See [this
repo](https://github.com/pytorch/examples/tree/master/word_language_model) and
[the
doc](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
for reference.

The code for the masked_cross_entropy was borrowed from [this
repo](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation).

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
-> This is no longer true, will work on implementation 1 sentence at teh time
and then optimize.


### Model input
The input to the encoder is a matrix of the tokens ID for each sequence
(columns is the batch axis). The embedding layer of pytorch maps the ID to an
embedding of the same dimensions as the hidden state.

### References

One of the first if not the first seq2seq learning, they talk about the
necessity of doing beam search. See *Decoding and Rescoring* [link to
paper](https://arxiv.org/pdf/1409.3215v1.pdf%3B)

Other paper about beam search strategies [link to
paper](https://arxiv.org/pdf/1702.01806.pdf)

Beam search implementation from [this repo](
https://github.com/GuessWhatGame/guesswhat/blob/master/src/guesswhat/models/qgen/qgen_beamsearch_wrapper.py)

Highly inspired but was modified to fit our needs and model

