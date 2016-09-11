# HNMT: the Helsinki Neural Machine Translation system

This is a neural network-based machine translation system developed
at the University of Helsinki.

It is currently rather experimental, but the user interface and setup
procedure should be simple enough for people to try out.

## Features

* biLSTM encoder which can be either character-based or hybrid word/character
  ([Luong & Manning 2016](http://arxiv.org/abs/1604.00788))
* LSTM decoder which can be either character-based or word-based
* Variational dropout ([Gal 2015](http://arxiv.org/abs/1512.05287))
  and Layer Normalization ([Ba et al. 2016](https://arxiv.org/abs/1607.06450))
* Bayesian word alignment model for guiding attention mechanism (experimental)

## Requirements

* A GPU if you plan to train your own models
* Python 3.4 or higher
* [Theano](http://deeplearning.net/software/theano/) (use the development
  version)
* [BNAS](https://github.com/robertostling/bnas)
* [NLTK](http://www.nltk.org/) for tokenization
* [efmaral](https://github.com/robertostling/efmaral) if you want to try the
  experimental supervised attention feature (see below)

## Quick start

If Theano and BNAS are installed, you should be able to simply run
`hnmt.py`. Run with the `--help` argument to see the available command-line
options.

Training a model on the Europarl corpus can be done like this:

    python3 hnmt.py --source europarl-v7.sv-en.en \
                    --target europarl-v7.sv-en.sv \
                    --source-tokenizer word \
                    --target-tokenizer char \
                    --source-vocabulary 50000 \
                    --max-source-length 30 \
                    --max-target-length 180 \
                    --batch-size 32 \
                    --training-time 24 \
                    --log en-sv.log \
                    --save-model en-sv.model

This will create a model with a hybrid encoder (with 50k vocabulary size and
character-level encoding for the rest) and character-based
decoder, filtering out sentences longer than 30 words (source) or 180
characters (target) and training for 24 hours. Development set cross-entropy
and some other statistics appended to this file, which is usually the best way
of monitoring training. Training loss and development set translations will be
written to stdout, so redirecting this or using `tee` is recommended.

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

## Using efmaral for attention supervision

Install [efmaral](https://github.com/robertostling/efmaral) and add it to your
`PYTHONPATH` or uncomment the `sys.path.append(...)` line in `hnmt.py`.
This is because `efmaral` does not yet have a proper installer (sorry).

Then you can simply add `--alignment-loss` when training to activate this
feature.

