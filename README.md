# HNMT: the Helsinki Neural Machine Translation system

This is a neural network-based machine translation system developed
at the University of Helsinki.

It is currently rather experimental, but the user interface and setup
procedure should be simple enough for people to try out.

## Updates

There has been a number of changes to the interface, due to a rewrite of the
data loading code so that not all training data is loaded into RAM. This
reduces memory consumption considerably.

* Training data is now given as a single file with source/target sentences
  separated by a ||| token, using the `--train` argument. The `--source` and
  `--target` arguments should not be used.
* Held-out sentences for training monitoring must be specified using
  `--heldout-source` and `--heldout-target` (as opposed to the training data,
  these must be contained in two separate files for the source and target
  language).
* Vocabularies must be computed in advance. There is a new tool,
  `make_encoder.py` which does this. One should be created for each of the
  source and target texts, and loaded with `--load-source-vocabulary` and
  `--load-target-vocabulary` respectively.
* The semantics of `--beam-budget` have changed a bit, but the acceptable
  values should be roughly the same as before, and depends on model size and
  GPU RAM but not on sentence length. `--batch-size` is only used during
  translation.
* You can use `--backwards yes` to train a model where all the input is
  reversed (on the character level). Currently the output is kept reversed,
  but this is subject to modification.

## Features

* biLSTM encoder which can be either character-based or hybrid word/character
  ([Luong & Manning 2016](http://arxiv.org/abs/1604.00788))
* LSTM decoder which can be either character-based or word-based
* Beam search with coverage penalty
  based on [Wu et al. (2016)](https://arxiv.org/pdf/1609.08144.pdf)).
* Partial support for byte pair encoding
  ([Sennrich et al. (2015)](https://arxiv.org/abs/1508.07909))
* Variational dropout ([Gal 2015](http://arxiv.org/abs/1512.05287))
  and Layer Normalization ([Ba et al. 2016](https://arxiv.org/abs/1607.06450))
* Context gates ([Tu et al. 2017](https://arxiv.org/pdf/1608.06043.pdf))

## Requirements

* A GPU if you plan to train your own models
* Python 3.4 or higher
* [Theano](http://deeplearning.net/software/theano/) (use the development
  version)
* [BNAS](https://github.com/robertostling/bnas)
* [NLTK](http://www.nltk.org/) for tokenization, but note that HNMT also
  supports pre-tokenized data from external tokenizers

## Quick start

If Theano and BNAS are installed, you should be able to simply run
`hnmt.py`. Run with the `--help` argument to see the available command-line
options.

Training a model on the Europarl corpus can be done like this:

    python3 make_encoder.py --min-char-count 2 --tokenizer word \
                            --hybrid --vocabulary 50000 \
                            --output vocab.sv europarl-v7.sv-en.sv

    python3 make_encoder.py --min-char-count 2 --tokenizer char \
                            --output vocab.en europarl-v7.sv-en.en

    python3 hnmt.py --train europarl-v7.sv-en \
                    --source-tokenizer word \
                    --target-tokenizer char \
                    --heldout-source dev.sv \
                    --heldout-target dev.en \
                    --load-source-vocabulary vocab.sv \
                    --load-target-vocabulary vocab.en \
                    --batch-budget 32 \
                    --training-time 24 \
                    --log en-sv.log \
                    --save-model en-sv.model

This will create a model with a hybrid encoder (with 50k vocabulary size and
character-level encoding for the rest) and character-based decoder, and train
it for 24 hours. Development set cross-entropy and some other statistics
appended to this file, which is usually the best way of monitoring training.
Training loss and development set translations will be written to stdout, so
redirecting this or using `tee` is recommended.

Note that `--heldout-source` and `--heldout-target` are mandatory, and that
while the training data contains sentence pairs separated by ||| in the same
file, the heldout sentences (which are only used for monitoring during
training) are separated into two files.

The resulting model can be used like this:

    python3 hnmt.py --load-model en-sv.model \
                    --translate test.en --output test.sv \
                    --beam-size 10

Note that when training a model from scratch, parameters can be set on the
commandline or otherwise the hard-coded defaults are ued. When continuing
training or doing translation (i.e. whenever the ``--load-model`` argument is
used), the defaults are encoded in the given model file, although some of
these (that do not change the network structure) can still be overridden by
commandline arguments.

For instance, the model above will assume that input files need to be
tokenized, but passing a pre-tokenized (space-separated) input can be done as
follows:

    python3 hnmt.py --load-model en-sv.model \
                    --translate test.en --output test.sv \
                    --source-tokenizer space \
                    --beam-size 10

## Resuming training

You can resume training by adding the `--load-model` argument without using
`--translate` (which disables training). For instance, if you want to keep
training the model above for another 48 hours on the same data:

    python3 hnmt.py --load-model en-sv.model
                    --training-time 48 \
                    --save-model en-sv-72h.model

## Segmentation

Select the tokenizer among these options:

* space: pre-segmented with spaces as separators
* char: split into character sequences
* word: use wordpunct from nltk
* bpe: pre-segmented with BPE (remove '@@ ' from final output)

TODO: support BPE as internal segmentation (apply_bpe to training data)

## Log files

During training, the `*.log` file reports the following information (in order, one column per item):
* Seconds since start
* Average cross-entropy per symbol
* Attention cross-entropy (values are sensible only when using attention supervision)
* Seconds used for test sentences
* Number of processed batches
* Number of processed sentences

The `*.log.eval` file reports evaluation metrics on the heldout set (in order, one column per item):
* Seconds since start
* BLEU score
* chrF score
* Number of processed batches
* Number of processed sentences
