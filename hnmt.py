"""HNMT: Helsinki Neural Machine Translation system.

See README.md for further documentation.
"""

import gzip
import sys
import os
import inspect
import random
from pprint import pprint

# add the path to hnmt to the system path to import BLEU etc
cmd_folder = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from hnmt.bleu import BLEU
from hnmt.chrF import chrF
from hnmt.bpe import BPE

try:
    from nltk import word_tokenize, wordpunct_tokenize
except ImportError:
    print('HNMT: WARNING: NLTK not installed, will not be able to use '
          'internal tokenizer', file=sys.stderr, flush=True)
#from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
#from nltk.translate.chrf_score import corpus_chrf

import numpy as np
import theano
from theano import tensor as T

# encoding in advance
# FIXME: either merge this into bnas, fork bnas, or make hnmt a proper package
from text import TextEncoder, Encoded
from search import beam_with_coverage
from largetext import ShuffledText, HalfSortedIterator

from bnas.model import Model, Linear, Embeddings, LSTMSequence
try:
    from bnas.model import ContextGateSequence
except ImportError:
    print('HNMT: can not import ContextGateSequence from BNAS, please upgrade'
          ' if you want to use context gates',
          file=sys.stderr, flush=True)
from bnas.optimize import Adam, iterate_batches
from bnas.init import Gaussian
from bnas.utils import softmax_3d
from bnas.loss import batch_sequence_crossentropy
from bnas.fun import function

def combo_len(src_weight, tgt_weight, x_weight):
    def _combo_len(pair):
        src, tgt = len(pair[0].sequence), len(pair[1].sequence)
        return (  (src * src_weight)
                + (tgt * tgt_weight)
                + (src * tgt * x_weight))
    return _combo_len

def local_sort(data, len_f, sort_size=16*32):
    order = list(range(len(data)))
    random.shuffle(order)
    reverse = False
    for i in range(0, len(data), sort_size):
        subset = [data[j] for j in order[i:i + sort_size]]
        subset.sort(key=len_f, reverse=reverse)
        for sent in subset:
            yield sent
        # alternate between sort directions,
        # to avoid discontinuity causing a minibatch with very long
        # and very short sentences
        reverse = not reverse

def iterate_variable_batches(data, batch_budget, len_f,
                             const_weight=0, src_weight=0,
                             tgt_weight=1, x_weight=0, c_weight=0,
                             sort_size=16*32):
    """Iterate over minibatches.

    Produces variable-size minibatches,
    increasing the minibatch size if sentences are short.
    This differs from the version in bnas,
    from which this function is derived.

    Arguments
    ---------
    data : list of data items (encoded example/label pairs)
        Data set to iterate over
    batch_budget : float
        Maximum number of "budget units" to include in a minibatch
    len_f : function
        A function mapping items from the
        data array to some ordered type. sort_size sentences will be randomly
        sampled at a time, the examples inside sorted and cut up into batches.
        This is useful for variable-length sequences, so that batches aren't
        too sparse.
    src_weight : float
        How many "budget units" does an increase of one source token cost.
    tgt_weight : float
        How many "budget units" does an increase of one target token cost.
    x_weight : float
        A cost in "budget units" for the product of source and target lengths.
        Useful e.g. for attention.
    c_weight : float
        A cost in "budget units" for each character in the unknown tokens.
    sort_size : int
        How many sentences to sample for sorting.
    """
    def within_budget(n, src, tgt, chars):
        cost = n * (const_weight
                 + (src * src_weight)
                 + (tgt * tgt_weight)
                 + (src * tgt * x_weight)
                 + (chars * c_weight))
        return cost < batch_budget

    minibatch = []
    max_src_len = 0
    max_tgt_len = 0
    tot_unk_n = 0
    max_unk_len = 0
    for sent in local_sort(data, len_f, sort_size=sort_size):
        src_len, tgt_len = len(sent[0].sequence), len(sent[1].sequence)
        unk_n = len(sent[0].unknown)
        if sent[0].unknown:
            unk_len = max(len(x) for x in sent[0].unknown)
        else:
            unk_len = 0
        if within_budget(len(minibatch) + 1,
                         max(max_src_len, src_len),
                         max(max_tgt_len, tgt_len),
                         (tot_unk_n + unk_n) * max(max_unk_len, unk_len)
                        ):
            minibatch.append(sent)
            max_src_len = max(max_src_len, src_len)
            max_tgt_len = max(max_tgt_len, tgt_len)
            tot_unk_n += unk_n
            max_unk_len = max(max_unk_len, unk_len)
        else:
            yield minibatch
            # start a new minibatch containing rejected sentence
            minibatch = [sent]
            max_src_len = src_len
            max_tgt_len = tgt_len
            tot_unk_n = unk_n
            max_unk_len = unk_len
    # final incomplete minibatch
    yield minibatch


class NMT(Model):
    def __init__(self, name, config):
        super().__init__(name)
        self.config = config

        pprint(config)
        sys.stdout.flush()

        self.add(Embeddings(
            'src_char_embeddings',
            len(config['src_encoder'].sub_encoder),
            config['src_char_embedding_dims'],
            dropout=config['char_embeddings_dropout']))

        self.add(Embeddings(
            'src_embeddings',
            len(config['src_encoder']),
            config['src_embedding_dims'],
            dropout=config['embeddings_dropout']))

        self.add(Embeddings(
            'trg_embeddings',
            len(config['trg_encoder']),
            config['trg_embedding_dims']))

        self.add(Linear(
            'hidden',
            config['decoder_state_dims'],
            config['trg_embedding_dims'],
            dropout=config['dropout'],
            layernorm=config['layernorm']))

        self.add(Linear(
            'emission',
            config['trg_embedding_dims'],
            len(config['trg_encoder']),
            w=self.trg_embeddings._w.T))

        self.add(Linear(
            'proj_h0',
            config['encoder_state_dims'],
            config['decoder_state_dims'],
            dropout=config['dropout'],
            layernorm=config['layernorm']))

        if config['decoder_gate'] == 'lstm':
            self.add(Linear(
                'proj_c0',
                config['encoder_state_dims'],
                config['decoder_state_dims'],
                dropout=config['dropout'],
                layernorm=config['layernorm']))

        # The total loss is
        #   lambda_o*xent(target sentence) + lambda_a*xent(alignment)
        self.lambda_o = theano.shared(
                np.array(1.0, dtype=theano.config.floatX))
        self.lambda_a = theano.shared(
                np.array(config['alignment_loss'], dtype=theano.config.floatX))
        for prefix, backwards in (('fwd', False), ('back', True)):
            self.add(LSTMSequence(
                prefix+'_char_encoder', backwards,
                config['src_char_embedding_dims'] + (
                    (config['src_embedding_dims'] // 2) if backwards else 0),
                config['src_embedding_dims'] // 2,
                layernorm=config['encoder_layernorm'],
                dropout=config['recurrent_dropout'],
                trainable_initial=True,
                offset=0))
        for prefix, backwards in (('fwd', False), ('back', True)):
            self.add(LSTMSequence(
                prefix+'_encoder', backwards,
                config['src_embedding_dims'] + (
                    config['encoder_state_dims'] if backwards else 0),
                config['encoder_state_dims'],
                layernorm=config['encoder_layernorm'],
                dropout=config['recurrent_dropout'],
                trainable_initial=True,
                offset=0))
        decoder_class = ContextGateSequence \
                if config['decoder_gate'] == 'context' \
                else LSTMSequence
        self.add(decoder_class(
            'decoder', False,
            config['trg_embedding_dims'],
            config['decoder_state_dims'],
            layernorm=config['decoder_layernorm'],
            dropout=config['recurrent_dropout'],
            attention_dims=config['attention_dims'],
            attended_dims=2*config['encoder_state_dims'],
            trainable_initial=False,
            offset=-1))

        h_t = T.matrix('h_t')
        self.predict_fun = function(
                [h_t],
                T.nnet.softmax(self.emission(T.tanh(self.hidden(h_t)))))

        inputs = T.lmatrix('inputs')
        inputs_mask = T.bmatrix('inputs_mask')
        chars = T.lmatrix('chars')
        chars_mask = T.bmatrix('chars_mask')
        outputs = T.lmatrix('outputs')
        outputs_mask = T.bmatrix('outputs_mask')
        attention = T.tensor3('attention')

        self.x = [inputs, inputs_mask, chars, chars_mask]
        self.y = [outputs, outputs_mask, attention]

        self.encode_fun = function(self.x, self.encode(*self.x))
        self.xent_fun = function(self.x+self.y, self.xent(*(self.x+self.y)))
        self.pred_fun = function(self.x+self.y[:-1], self(*(self.x+self.y[:-1])))

        # stats
        #self.beam_ends = np.zeros((config['max_target_length'],))

    def xent(self, inputs, inputs_mask, chars, chars_mask,
             outputs, outputs_mask, attention):
        pred_outputs, pred_attention = self(
                inputs, inputs_mask, chars, chars_mask, outputs, outputs_mask)
        outputs_xent = batch_sequence_crossentropy(
                pred_outputs, outputs[1:], outputs_mask[1:])
        # Note that pred_attention will contain zero elements for masked-out
        # character positions, to avoid trouble with log() we add 1 for zero
        # element of attention (which after multiplication will be removed
        # anyway).
        batch_size = attention.shape[1].astype(theano.config.floatX)
        attention_mask = (inputs_mask.dimshuffle('x', 1, 0) *
                          outputs_mask[1:].dimshuffle(0, 1, 'x')
                          ).astype(theano.config.floatX)
        epsilon = 1e-6
        attention_xent = (
                   -attention[1:]
                 * T.log(epsilon + pred_attention + (1-attention_mask))
                 * attention_mask).sum() / batch_size
        return outputs_xent, attention_xent

    def loss(self, *args):
        outputs_xent, attention_xent = self.xent(*args)
        return super().loss() + self.lambda_o*outputs_xent \
                + self.lambda_a*attention_xent

    def unify_embeddings(self, model):
        """Ensure that the embeddings use the same vocabulary as model"""
        other_src_char_encoder = model.config['src_encoder'].sub_encoder
        other_src_encoder = model.config['src_encoder']
        other_trg_encoder = model.config['trg_encoder']
        src_char_encoder = self.config['src_encoder'].sub_encoder
        src_encoder = self.config['src_encoder']
        trg_encoder = self.config['trg_encoder']

        def make_translation(this, that):
            return np.array([this.index[x] for x in this.vocab])

        if src_char_encoder.vocab != other_src_char_encoder.vocab:
            trans_src_char = make_translation(
                    src_char_encoder, other_src_char_encoder)
            self.src_char_embeddings._w.set_value(
                    self.src_char_embeddings._w.get_value()[trans_src_char])

        if src_encoder.vocab != other_src_encoder.vocab:
            trans_src = make_translation(src_encoder, other_src_encoder)
            self.src_embeddings._w.set_value(
                    self.src_embeddings._w.get_value()[trans_src])

        if trg_encoder.vocab != other_trg_encoder.vocab:
            trans_trg = make_translation(trg_encoder, other_trg_encoder)
            self.trg_embeddings._w.set_value(
                    self.trg_embeddings._w.get_value()[trans_trg])


    def search(self, inputs, inputs_mask, chars, chars_mask,
               max_length, beam_size=8,
               alpha=0.2, beta=0.2, gamma=1.0, len_smooth=5.0, others=[],
               **kwargs):
        # list of models in the ensemble
        models = [self] + others
        n_models = len(models)
        if self.config['decoder_gate'] == 'lstm':
            n_states = 2
        else:
            n_states = 1

        # tuple (h_0, c_0, attended) for each model in the ensemble
        models_init = [m.encode_fun(inputs, inputs_mask, chars, chars_mask)
                       for m in models]

        # precomputed sequences for attention, one for each model
        models_attended_dot_u = [
                m.decoder.attention_u_fun()(model_init[-1])
                for m, model_init in zip(models, models_init)]

        # output embeddings for each model
        models_embeddings = [
                m.trg_embeddings._w.get_value(borrow=False)
                for m in models]


        def step(i, states, outputs, outputs_mask, sent_indices):
            if self.config['decoder_gate'] == 'lstm':
                models_result = [
                        models[idx].decoder.step_fun()(
                            models_embeddings[idx][outputs[-1]],
                            states[idx*n_states+0],
                            states[idx*n_states+1],
                            models_init[idx][-1][:,sent_indices,...],
                            models_attended_dot_u[idx][:,sent_indices,...],
                            inputs_mask[:,sent_indices])
                        for idx in range(n_models)]
            else:
                models_result = [
                        models[idx].decoder.step_fun()(
                            models_embeddings[idx][outputs[-1]],
                            states[idx*n_states+0],
                            models_init[idx][-1][:,sent_indices,...],
                            models_attended_dot_u[idx][:,sent_indices,...],
                            inputs_mask[:,sent_indices])
                        for idx in range(n_models)]
            mean_attention = np.array(
                    [models_result[idx][-1] for idx in range(n_models)]
                 ).mean(axis=0)
            models_predict = np.array(
                    [models[idx].predict_fun(models_result[idx][0])
                     for idx in range(n_models)])
            dist = models_predict.mean(axis=0)
            return ([x for result in models_result for x in result[:n_states]],
                    dist, mean_attention)

        initial = [x for h_0, c_0, _ in models_init for x in [h_0, c_0]] \
                if self.config['decoder_gate'] == 'lstm' \
                else [x for h_0, _ in models_init for x in [h_0,]]
        result, i = beam_with_coverage(
                step,
                initial,
                models_init[0][0].shape[0],
                self.config['trg_encoder']['<S>'],
                self.config['trg_encoder']['</S>'],
                max_length,
                inputs_mask,
                beam_size=beam_size,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                len_smooth=len_smooth,
                **kwargs)
        #self.beam_ends[i] += 1
        return result

    #def search_single(self, inputs, inputs_mask, chars, chars_mask, max_length,
    #           beam_size=8):
    #    h_0, c_0, attended = self.encode_fun(
    #            inputs, inputs_mask, chars, chars_mask)
    #    return self.decoder.search(
    #            self.predict_fun,
    #            self.trg_embeddings._w.get_value(borrow=True),
    #            self.config['trg_encoder']['<S>'],
    #            self.config['trg_encoder']['</S>'],
    #            max_length,
    #            h_0=h_0, c_0=c_0,
    #            attended=attended,
    #            attention_mask=inputs_mask,
    #            beam_size=beam_size)

    def encode(self, inputs, inputs_mask, chars, chars_mask):
        # First run a bidirectional LSTM encoder over the unknown word
        # character sequences.
        embedded_chars = self.src_char_embeddings(chars)
        fwd_char_h_seq, fwd_char_c_seq = self.fwd_char_encoder(
                embedded_chars, chars_mask)
        back_char_h_seq, back_char_c_seq = self.back_char_encoder(
                T.concatenate([embedded_chars, fwd_char_h_seq], axis=-1),
                chars_mask)

        # Concatenate the final states of the forward and backward character
        # encoders. These form a matrix of size:
        #   n_chars x src_embedding_dims
        # NOTE: the batch size here is n_chars, which is the total number of
        # unknown words in all the sentences in the inputs matrix.
        # Create an empty matrix if there are no unknown words
        # (e.g. pure word-level encoder)
        char_vectors = theano.ifelse.ifelse(T.gt(chars.shape[0], 0),
                T.concatenate([fwd_char_h_seq[-1], back_char_h_seq[0]], axis=-1),
                T.zeros([0, self.config['src_embedding_dims']],
                dtype=theano.config.floatX))

        # Compute separate masks for known words (with input symbol >= 0)
        # and unknown words (with input symbol < 0).
        known_mask = inputs_mask * T.ge(inputs, 0)
        unknown_mask = inputs_mask * T.lt(inputs, 0)
        # Split the inputs matrix into two, one indexing unknown words (from
        # the char_vectors matrix) and the other known words (from the source
        # word embeddings).
        unknown_indexes = (-inputs-1) * unknown_mask
        known_indexes = inputs * known_mask

        # Compute the final embedding sequence by mixing the known word
        # vectors with the character encoder output of the unknown words.
        # If there is no character encoder, just use the known word vectors.
        embedded_unknown = char_vectors[unknown_indexes]
        embedded_known = self.src_embeddings(known_indexes)
        embedded_inputs = theano.ifelse.ifelse(T.gt(chars.shape[0], 0),
                (unknown_mask.dimshuffle(0,1,'x').astype(
                    theano.config.floatX) * embedded_unknown) + \
                (known_mask.dimshuffle(0,1,'x').astype(
                    theano.config.floatX) * embedded_known),
                known_mask.dimshuffle(0,1,'x').astype(
                    theano.config.floatX) * embedded_known)

        # Forward encoding pass
        fwd_h_seq, fwd_c_seq = self.fwd_encoder(embedded_inputs, inputs_mask)
        # Backward encoding pass, using hidden states from forward encoder
        back_h_seq, back_c_seq = self.back_encoder(
                T.concatenate([embedded_inputs, fwd_h_seq], axis=-1),
                inputs_mask)
        # Initial states for decoder
        h_0 = T.tanh(self.proj_h0(back_h_seq[0]))
        if self.config['decoder_gate'] == 'lstm':
            c_0 = T.tanh(self.proj_c0(back_c_seq[0]))
        # Attention on concatenated forward/backward sequences
        attended = T.concatenate([fwd_h_seq, back_h_seq], axis=-1)
        if self.config['decoder_gate'] == 'lstm':
            return h_0, c_0, attended
        else:
            return h_0, attended

    def __call__(self, inputs, inputs_mask, chars, chars_mask,
                 outputs, outputs_mask):
        embedded_outputs = self.trg_embeddings(outputs)
        if self.config['decoder_gate'] == 'lstm':
            h_0, c_0, attended = self.encode(
                    inputs, inputs_mask, chars, chars_mask)
            h_seq, c_seq, attention_seq = self.decoder(
                    embedded_outputs, outputs_mask, h_0=h_0, c_0=c_0,
                    attended=attended, attention_mask=inputs_mask)
        else:
            h_0, attended = self.encode(
                    inputs, inputs_mask, chars, chars_mask)
            h_seq, attention_seq = self.decoder(
                    embedded_outputs, outputs_mask, h_0=h_0,
                    attended=attended, attention_mask=inputs_mask)
        pred_seq = softmax_3d(self.emission(T.tanh(self.hidden(h_seq))))

        return pred_seq, attention_seq

    def create_optimizer(self):
        return Adam(
                self.parameters(),
                self.loss(*(self.x + self.y)),
                self.x, self.y,
                grad_max_norm=5.0)

    def average_parameters(self, others):
        for name, p in self.parameters():
            p.set_value(np.mean(
                    [p.get_value(borrow=True)] + \
                    [other.parameter(name).get_value(borrow=True)
                     for other in others],
                    axis=0))

# TODO: make it possible to apply BPE here
# For the moment, bpe tokenizer == space tokenizer

def read_sents(filename, tokenizer, lower):
    def process(line):
        if lower: line = line.lower()
        if tokenizer == 'char': return line.strip()
        elif tokenizer == 'word': return word_tokenize(line)
        return line.split()
    if filename.endswith('.gz'):
        def open_func(fname):
            return gzip.open(fname, 'rt', encoding='utf-8')
    else:
        def open_func(fname):
            return open(fname, 'r', encoding='utf-8')
    with open_func(filename) as f:
        return list(map(process, f))
    
def detokenize(sent, tokenizer):
    if tokenizer == 'bpe':
        string = ' '.join(sent)
        return string.replace("@@ ", "")
    return ('' if tokenizer == 'char' else ' ').join(sent)


def main():
    import argparse
    import pickle
    import sys
    import os.path
    from time import time

    parser = argparse.ArgumentParser(
            description='HNMT -- Helsinki Neural Machine Translation system')

    parser.add_argument('--load-model', type=str,
            metavar='FILE(s)',
            help='name of the model file(s) to load from, comma-separated list'
                 ' of colon-separated lists. The colon-separated lists should'
                 ' be savepoints from the same training run, the outer-level'
                 ' comma-separated list is of different models to ensemble.')
    parser.add_argument('--load-submodel', type=str,
            metavar='FILE(s)',
            help='name of the submodel file(s) to load from, comma-separated list of modelname=file')
    parser.add_argument('--save-model', type=str,
            metavar='FILE',
            help='name of the model file to save to')
    parser.add_argument('--split-model', type=str,
            metavar='FILE',
            help='split an existing model into separate files for each submodule')
    parser.add_argument('--ensemble-average', action='store_true',
            help='ensemble models by averaging parameters (DEPRECATED)')
    parser.add_argument('--translate', type=str,
            metavar='FILE',
            help='name of file to translate')
    parser.add_argument('--nbest-list', type=int,
            default=0,
            metavar='N',
            help='print n-best list in translation model')
    parser.add_argument('--reference', type=str,
            metavar='FILE',
            help='name of the reference translations file')
    parser.add_argument('--output', type=str,
            metavar='FILE',
            help='name of file to write translated text to')
    parser.add_argument('--decoder-gate', type=str,
            choices=('lstm','context'),
            metavar='FILE', default=argparse.SUPPRESS,
            help='type of decoder gate ("lstm" or "context")')
    parser.add_argument('--heldout-source', type=str,
            metavar='FILE', default=argparse.SUPPRESS,
            help='name of test-set file (source language)')
    parser.add_argument('--heldout-target', type=str,
            metavar='FILE', default=argparse.SUPPRESS,
            help='name of test-set file (target language)')
    parser.add_argument('--beam-size', type=int, default=argparse.SUPPRESS,
            metavar='N',
            help='beam size during translation')
    parser.add_argument('--alpha', type=float, default=argparse.SUPPRESS,
            metavar='X',
            help='length penalty weight during beam translation')
    parser.add_argument('--beta', type=float, default=argparse.SUPPRESS,
            metavar='X',
            help='coverage penalty weight during beam translation')
    parser.add_argument('--gamma', type=float, default=argparse.SUPPRESS,
            metavar='X',
            help='overattention penalty weight during beam translation')
    parser.add_argument('--len-smooth', type=float, default=argparse.SUPPRESS,
            metavar='X',
            help='smoothing constant for length penalty during beam translation')
    parser.add_argument('--save-every', type=int, default=argparse.SUPPRESS,
            metavar='N',
            help='save model every N training batches')
    parser.add_argument('--test-every', type=int, default=argparse.SUPPRESS,
            metavar='N',
            help='compute test set cross-entropy every N training batches')
    parser.add_argument('--translate-every', type=int,
            metavar='N',
            default=argparse.SUPPRESS,
            help='translate test set every N training batches')
    parser.add_argument('--train', type=str, default=argparse.SUPPRESS,
            metavar='FILE',
            help='name of training data file (with source ||| target pairs)')
    parser.add_argument('--score-source', type=str, default=argparse.SUPPRESS,
            metavar='FILE',
            help='name of source language file for sentence scoring')
    parser.add_argument('--score-target', type=str, default=argparse.SUPPRESS,
            metavar='FILE',
            help='name of target language test file for sentence scoring')
    parser.add_argument('--source-tokenizer', type=str,
            choices=('word', 'space', 'char', 'bpe'), default=argparse.SUPPRESS,
            help='type of preprocessing for source text')
    parser.add_argument('--target-tokenizer', type=str,
            choices=('word', 'space', 'char', 'bpe'), default=argparse.SUPPRESS,
            help='type of preprocessing for target text')
    #parser.add_argument('--max-source-length', type=int,
    #        metavar='N',
    #        default=argparse.SUPPRESS,
    #        help='maximum length of source sentence '
    #             '(unit given by --source-tokenizer)')
    parser.add_argument('--max-target-length', type=int, default=1000,
            metavar='N',
            help='maximum length of target sentence during translation '
                 '(unit given by --target-tokenizer)')
    parser.add_argument('--batch-size', type=int, default=argparse.SUPPRESS,
            metavar='N',
            help='minibatch size of devset (FIXME)')
    parser.add_argument('--batch-budget', type=float, default=argparse.SUPPRESS,
            metavar='X',
            help='minibatch budget during training. '
                 'The optimal value depends on model size and available GPU memory. '
                 'Try values between 20 and 200')
    parser.add_argument('--log-file', type=str,
            metavar='FILE',
            help='name of training log file')
    parser.add_argument('--source-lowercase', type=str, choices=('yes','no'),
            default=argparse.SUPPRESS,
            help='convert source text to lowercase before processing')
    parser.add_argument('--target-lowercase', type=str, choices=('yes','no'),
            default=argparse.SUPPRESS,
            help='convert target text to lowercase before processing')
    parser.add_argument('--load-source-vocabulary', type=str, default=None,
            metavar='FILE',
            help='load source vocabulary from this file (created by '
                 'make_encoder.py). This should not be combined with '
                 '--load-model, since that already loads the vocabulary '
                 'stored in the model file')
    parser.add_argument('--load-target-vocabulary', type=str, default=None,
            metavar='FILE',
            help='load target vocabulary from this file (created by '
                 'make_encoder.py). This should not be combined with '
                 '--load-model, since that already loads the vocabulary '
                 'stored in the model file')
    parser.add_argument('--source-vocabulary', type=int, default=10000,
            metavar='N',
            help='maximum size of source vocabulary')
    parser.add_argument('--target-vocabulary', type=int, default=None,
            metavar='N',
            help='maximum size of target vocabulary')
    parser.add_argument('--min-char-count', type=int,
            metavar='N',
            help='drop all characters with count < N in training data')
    parser.add_argument('--dropout', type=float, default=0.0,
            metavar='FRACTION',
            help='use dropout for non-recurrent connections '
                 'with the given factor')
    parser.add_argument('--recurrent-dropout', type=float, default=0.0,
            metavar='FRACTION',
            help='use dropout for recurrent connections with the given factor')
    parser.add_argument('--layer-normalization', action='store_true',
            help='use layer normalization')
    parser.add_argument('--word-embedding-dims', type=int, default=256,
            metavar='N',
            help='size of word embeddings')
    parser.add_argument('--char-embedding-dims', type=int, default=64,
            metavar='N',
            help='size of character embeddings')
    parser.add_argument('--target-embedding-dims', type=int, default=None,
            metavar='N',
            help='size of target embeddings '
            '(default: size of input word or char embedding')
    parser.add_argument('--encoder-state-dims', type=int, default=256,
            metavar='N',
            help='size of encoder state')
    parser.add_argument('--decoder-state-dims', type=int, default=512,
            metavar='N',
            help='size of decoder state')
    parser.add_argument('--attention-dims', type=int, default=256,
            metavar='N',
            help='size of attention vectors')
    parser.add_argument('--alignment-loss', type=float, default=0.0,
            metavar='X',
            help='alignment cross-entropy contribution to loss function '
                 '(DEPRECATED)')
    parser.add_argument('--alignment-decay', type=float, default=0.9999,
            metavar='X',
            help='decay factor of alignment cross-entropy contribution '
                 '(DEPRECATED)')
    parser.add_argument('--learning-rate', type=float, default=None,
            metavar='X',
            help='override the default learning rate for optimizer with X')
    parser.add_argument('--training-time', type=float, default=24.0,
            metavar='HOURS',
            help='training time')
    parser.add_argument('--random-seed', type=int, default=123,
            metavar='N',
            help='random seed for repeatable sorting of data')
    parser.add_argument('--source-bpe-codes', type=str,
            metavar='FILE',
            help='name of source language BPE codes file (and apply them)')
    parser.add_argument('--target-bpe-codes', type=str,
            metavar='FILE',
            help='name of target language BPE codes file (and apply them)')
    parser.add_argument('--score', type=str,
            metavar='FILE',
            help='score the sentence pairs defined by --test-target and '
                 '--test-source and write scores to this file')

    args = parser.parse_args()
    args_vars = vars(args)

    random.seed(args.random_seed)

    overridable_options = {
            'save_every': 1000,
            'test_every': 25,
            'translate_every': 250,
            'batch_size': 32,
            'batch_budget': 32,
            'source_lowercase': 'no',
            'target_lowercase': 'no',
            'source_tokenizer': 'space',
            'target_tokenizer': 'char',
            'train': None,
            'decoder_gate': 'lstm',
            'heldout_source': None,
            'heldout_target': None,
            'beam_size': 8,
            'alpha': 0.01,
            'beta': 0.4,
            'gamma': 1.0,
            'len_smooth': 5.0,}

    # read and use byte-pair encodings
    # TODO: the option doesn't work yet (change read_sents)
    # TODO: should we store BPE codes in the model file?

    srcbpe = False
    trgbpe = False

    if args.source_bpe_codes:
        srcbpe_codes = BPE(args.source_bpe_codes)
        srcbpe = True

    if args.target_bpe_codes:
        trgbpe_codes = BPE(args.source_bpe_codes)
        trgbpe = True

    if args.translate:
        models = []
        configs = []
        if ':' in args.load_model:
            print('HNMT: will average model savepoints',
                  file=sys.stderr, flush=True)
        if ',' in args.load_model:
            print('HNMT: will ensemble separate models',
                  file=sys.stderr, flush=True)
        # We may want to ensemble several independent models (modelA,
        # modelB, ...) but also average the parameters from some savepoints
        # within each model (modelA.1, modelA.2, ...).
        # This would be encoded as:
        #   --load-model modelA.1:modelA.2,modelB.1:modelB.2
        for group_filenames in args.load_model.split(','):
            group_models = []
            group_configs = []
            for filename in group_filenames.split(':'):
                print('HNMT: loading ensemble part %s...' % filename,
                      file=sys.stderr, flush=True)
                with open(filename, 'rb') as f:
                    group_configs.append(pickle.load(f))
                    group_models.append(NMT('nmt', group_configs[-1]))
                    group_models[-1].load(f)
            models.append(group_models[0])
            configs.append(group_configs[0])
            if len(group_models) > 1:
                models[-1].average_parameters(group_models[1:])

        model = models[0]
        config = configs[0]
        # allow loading old models without these parameters
        if 'alpha' not in config:
            config['alpha'] = 0.01
        if 'beta' not in config:
            config['beta'] = 0.4
        if 'gamma' not in config:
            config['gamma'] = 1.0
        if 'len_smooth' not in config:
            config['len_smooth'] = 5.0
        for c in configs[1:]:
            assert c['trg_encoder'].vocab == config['trg_encoder'].vocab

        for option in overridable_options:
            if option in args_vars: config[option] = args_vars[option]

    # split a modelfile into submodel files
    # (NOTE: this also saves the config for the whole model)
    elif args.split_model:
        if args.load_model:
            with open(args.load_model, 'rb') as f:
                config = pickle.load(f)
                model = NMT('nmt', config)
                model.load(f)
            filebase = args.split_model
            for submodel in model.submodels.values():
                filename = filebase + '.' + submodel.name
                print('save submodel %s' % (filename),
                      file=sys.stderr, flush=True)
                with open(filename, 'wb') as f:
                    pickle.dump(config, f)
                    submodel.save(f)
        else:
            quit('Use --load-model to specify model to be split!');

    else:
        print('HNMT: starting training...', file=sys.stderr, flush=True)
        if args.load_model:
            with open(args.load_model, 'rb') as f:
                config = pickle.load(f)
                # allow loading old models without these parameters
                if 'alignment_decay' not in config:
                    config['alignment_decay'] = 0.9995
                if 'alpha' not in config:
                    config['alpha'] = 0.2
                if 'beta' not in config:
                    config['beta'] = 0.2
                if 'gamma' not in config:
                    config['gamma'] = 1.0
                if 'len_smooth' not in config:
                    config['len_smooth'] = 5.0
                model = NMT('nmt', config)
                model.load(f)

                ## overwrite some submodels if specified on command-line
                ## TODO: do I need this here or is enough to overwrite 
                ## model parameters later after creating the optimizer?
                #if args.load_submodel:
                #    for filespec in args.load_submodel.split(','):
                #        modelname,filename = filespec.split('=')
                #        with open(filename, 'rb') as f:
                #            print('HNMT: loading submodel %s from %s ...' % (modelname,filename),
                #                  file=sys.stderr, flush=True)
                #            ## TODO: should check that the submodel config is compatible with model config
                #            submodel_config = pickle.load(f)
                #            getattr(model,modelname).load(f)

                models = [model]
                optimizer = model.create_optimizer()
                if args.learning_rate:
                    optimizer.learning_rate = args.learning_rate
                ## load optimizer states unless there are submodels to be loaded later
                if not args.load_submodel:
                    optimizer.load(f)
            if not args.score:
                print('Continuing training from update %d...'%optimizer.n_updates,
                      flush=True)
            for option in overridable_options:
                if option in args_vars: config[option] = args_vars[option]

        else:
            assert args.save_model
            assert not os.path.exists(args.save_model)
            config = {}
            for option, default in overridable_options.items():
                config[option] = args_vars.get(option, default)

        if args.score:
            print('Load sentences for scoring ...', file=sys.stderr, flush=True)
            src_sents = read_sents(
                    config['score_source'], config['source_tokenizer'],
                    config['source_lowercase'] == 'yes')
            trg_sents = read_sents(
                    config['score_target'], config['target_tokenizer'],
                    config['target_lowercase'] == 'yes')

            assert len(src_sents) == len(trg_sents)
            with open(args.score, 'w') as outf:
                for i in range(0, len(src_sents), config['batch_size']):
                    src_batch = [
                            config['src_encoder'].encode_sequence(sent)
                            for sent in src_sents[i:i+config['batch_size']]]
                    trg_batch = [
                            config['trg_encoder'].encode_sequence(sent)
                            for sent in trg_sents[i:i+config['batch_size']]]
                    x = config['src_encoder'].pad_sequences(src_batch)
                    y = config['trg_encoder'].pad_sequences(trg_batch)
                    y = y + (np.ones(y[0].shape + (x[0].shape[0],),
                                        dtype=theano.config.floatX),)
                    test_inputs, test_inputs_mask, _, _ = x
                    test_outputs, test_outputs_mask, test_attention = y
                    pred_y, pred_attention = model.pred_fun(*(x + y[:-1]))
                    idx1, idx2 = np.indices(test_outputs[1:].shape)
                    p_y = pred_y[idx1, idx2, test_outputs[1:]] + \
                            (1 - test_outputs_mask[1:])
                    # Length of each sentence
                    len_y = test_outputs_mask.sum(0) - 2
                    # Log-probability per sentence
                    log_p = (np.log(p_y) * test_outputs_mask[1:]).sum(0)
                    # Length penalty of each sentence
                    lp = np.power((config['len_smooth'] + len_y) /
                                    (1.0 + config['len_smooth']),
                                  config['alpha'])
                    # Coverage penalty
                    attention_trg_sum = (
                            test_outputs_mask[1:][...,None] *
                            pred_attention).sum(0)
                    cp = ((np.log((1 - test_inputs_mask.T) +
                                  np.minimum(attention_trg_sum, 1.0))
                            * test_inputs_mask.T).sum(1)
                            * config['beta'])
                    score = cp + (log_p / lp)
                    #print(np.hstack([score[:,None], cp[:,None], lp[:,None],
                    #                 log_p[:,None]]))
                    for x in score:
                        print(x, file=outf, flush=True)
            print('Scores written to %s, exiting...' % args.score,
                  file=sys.stderr, flush=True)
            return

        def get_tokenizer(name, lowercase):
            if name == 'char':
                if lowercase:
                    return (lambda s: list(s.strip().lower()))
                else:
                    return (lambda s: list(s.strip()))
            elif name == 'space':
                if lowercase:
                    return (lambda s: s.lower().split())
                else:
                    return str.split
            elif name == 'word':
                if lowercase:
                    return (lambda s: word_tokenize(s.lower()))
                else:
                    return word_tokenize
            else:
                raise ValueError('Unknown tokenizer: %s' % name)

        tokenize_src = get_tokenizer(
                config['source_tokenizer'],
                config['source_lowercase'] == 'yes')
        tokenize_trg = get_tokenizer(
                config['target_tokenizer'],
                config['target_lowercase'] == 'yes')

        def tokenize_src_trg(s):
            src, trg = s.split(' ||| ')
            return tokenize_src(src), tokenize_trg(trg)

        src_trg_f = open(config['train'], 'rb')
        src_trg_st = ShuffledText(
                src_trg_f,
                # Using the defaults should be OK, but these settings could
                # reduce memory further:
                #block_size=0x1000,
                #max_blocks=64,
                seed=args.random_seed)
        train_iter = HalfSortedIterator(
                iter(src_trg_st),
                max_area=config['batch_budget']*0x200,
                preprocess=tokenize_src_trg,
                length=lambda pair: sum(map(len, pair)))

        if not args.load_model:
            # Source encoder is a hybrid, with a character-based encoder for
            # rare words and a word-level decoder for the rest.
            print('Loading vocabularies...', file=sys.stderr, flush=True)
            if args.load_source_vocabulary:
                with open(args.load_source_vocabulary, 'rb') as f:
                    src_encoder = pickle.load(f)
                    if not src_encoder.sub_encoder:
                        print("No sub-encoder found in source vocabulary, "
                              "creating dummy...",
                            file=sys.stderr, flush=True)
                        src_encoder.sub_encoder = TextEncoder(
                                counts={"": 1}, min_count=0,
                            special=())
            else:
                raise NotImplementedError(
                        '--load-source-vocabulary required when training '
                        'from scratch')
            if args.load_target_vocabulary:
                with open(args.load_target_vocabulary, 'rb') as f:
                    trg_encoder = pickle.load(f)
            else:
                raise NotImplementedError(
                        '--load-target-vocabulary required when training '
                        'from scratch')
            print('...done', file=sys.stderr, flush=True)

            if not args.target_embedding_dims is None:
                trg_embedding_dims = args.target_embedding_dims
            else:
                trg_embedding_dims = (
                    args.char_embedding_dims
                    if config['target_tokenizer'] == 'char'
                    else args.word_embedding_dims)
            config.update({
                'src_encoder': src_encoder,
                'trg_encoder': trg_encoder,
                'src_embedding_dims': args.word_embedding_dims,
                'trg_embedding_dims': trg_embedding_dims,
                'src_char_embedding_dims': args.char_embedding_dims,
                'char_embeddings_dropout': args.dropout,
                'embeddings_dropout': args.dropout,
                'recurrent_dropout': args.recurrent_dropout,
                'dropout': args.dropout,
                'encoder_state_dims': args.encoder_state_dims,
                'decoder_state_dims': args.decoder_state_dims,
                'attention_dims': args.attention_dims,
                'layernorm': args.layer_normalization,
                'alignment_loss': args.alignment_loss,
                'alignment_decay': args.alignment_decay,
                # NOTE: there are serious stability issues when ba1 is used for
                #       the encoder, and still problems with large models when
                #       the encoder uses ba2 and the decoder ba1.
                #       Is there any stable configuration?
                'encoder_layernorm':
                    'ba2' if args.layer_normalization else False,
                'decoder_layernorm':
                    'ba2' if args.layer_normalization else False,
                })

            model = NMT('nmt', config)
            models = [model]
            optimizer = model.create_optimizer()
            if args.learning_rate:
                optimizer.learning_rate = args.learning_rate

        ## load submodel parameters (overwrite existing ones)
        if args.load_submodel:
            for filespec in args.load_submodel.split(','):
                modelname,filename = filespec.split('=')
                with open(filename, 'rb') as f:
                    print('HNMT: loading submodel %s from %s ...' % (modelname,filename),
                          file=sys.stderr, flush=True)
                    ## TODO: should check that the submodel config is compatible with model config
                    submodel_config = pickle.load(f)
                    getattr(model,modelname).load(f)

    # By this point a model has been created or loaded, so we can define a
    # convenience function to perform translation.
    def translate(sents, encode=False, nbest=0):
        for i in range(0, len(sents), config['batch_size']):
            batch_sents = sents[i:i+config['batch_size']]
            if encode:
                batch_sents = [config['src_encoder'].encode_sequence(sent)
                               for sent in batch_sents]
            x = config['src_encoder'].pad_sequences(batch_sents)
            beams = model.search(
                    *(x + (args.max_target_length,)),
                    beam_size=config['beam_size'],
                    alpha=config['alpha'],
                    beta=config['beta'],
                    gamma=config['gamma'],
                    len_smooth=config['len_smooth'],
                    others=models[1:],
                    prune=(nbest == 0))
            nbest = min(nbest, config['beam_size'])
            for batch_sent_idx, (_, beam) in enumerate(beams):
                lines = []
                for best in list(beam)[:max(1, nbest)]:
                    encoded = Encoded(best.history + (best.last_sym,), None)
                    hypothesis = detokenize(
                        config['trg_encoder'].decode_sentence(encoded),
                        config['target_tokenizer'])
                    if nbest > 0:
                        lines.append(' ||| '.join((
                            str(i+batch_sent_idx), hypothesis,
                            str(best.norm_score))))
                    else:
                        yield hypothesis
                if lines:
                    yield '\n'.join(lines)

    if args.translate:
        print('Translating...', file=sys.stderr, flush=True, end='')
        outf = sys.stdout if args.output is None else open(
                args.output, 'w', encoding='utf-8')
        sents = read_sents(
                args.translate,
                config['source_tokenizer'],
                config['source_lowercase'] == 'yes')

        if args.reference: hypotheses = []
        if args.nbest_list: nbest = args.nbest_list
        else: nbest = 0
        for i,sent in enumerate(translate(
                sents, encode=True, nbest=nbest)):
            print('.', file=sys.stderr, flush=True, end='')
            print(sent, file=outf, flush=True)
            if args.reference:
                if nbest:
                    hypotheses.append(sent.split('\n')[0].split(' ||| ')[1])
                else:
                    hypotheses.append(sent)
        print(' done!', file=sys.stderr, flush=True)
        if args.output:
            outf.close()

        # compute BLEU if reference file is given
        if args.reference:
            trg = read_sents(args.reference,
                             config['target_tokenizer'],
                             config['target_lowercase'] == 'yes')

            if config['target_tokenizer'] == 'char':
                system = [detokenize(wordpunct_tokenize(s),'space')
                          for s in hypotheses]
                reference = [detokenize(
                                word_tokenize(detokenize(s,'char')), 'space')
                             for s in trg]
                print('BLEU = %f (%f, %f, %f, %f, BP = %f)' % BLEU(
                    system,[reference]))
                print('chrF = %f (precision = %f, recall = %f)' % chrF(
                    reference,system))
            else:
                reference = [detokenize(s,config['target_tokenizer'])
                             for s in trg ]
                print('BLEU = %f (%f, %f, %f, %f, BP = %f)' % BLEU(
                    hypotheses,[reference]))
                print('chrF = %f (precision = %f, recall = %f)' % chrF(
                    reference,hypotheses))
    else:
        def prepare_batch(batch_pairs):
            src_batch, trg_batch, links_maps_batch = \
                    list(zip(*batch_pairs))
            x = config['src_encoder'].pad_sequences(src_batch)
            y = config['trg_encoder'].pad_sequences(trg_batch)
            y = y + (np.ones(y[0].shape + (x[0].shape[0],),
                                dtype=theano.config.floatX),)
            return x, y

        if config['heldout_source'] and config['heldout_target']:
            print('Load test set ...', file=sys.stderr, flush=True)
            test_src = read_sents(
                args.heldout_source, config['source_tokenizer'],
                config['source_lowercase'] == 'yes')
            test_trg = read_sents(
                args.heldout_target, config['target_tokenizer'],
                config['target_lowercase'] == 'yes')
            if len(test_src) > config['batch_size']:
                print('reduce test set to batch size', file=sys.stderr, flush=True)
                test_src = test_src[:config['batch_size']]
                test_trg = test_trg[:config['batch_size']]

            test_trg_unencoded = test_trg
            test_src = [config['src_encoder'].encode_sequence(sent)
                     for sent in test_src]
            test_trg = [config['trg_encoder'].encode_sequence(sent)
                     for sent in test_trg]
            test_links_maps = [(None, None, None)]*len(test_src)
            test_pairs = list(zip(test_src, test_trg, test_links_maps))
        else:
            raise NotImplementedError(
                    'Heldout training sentences is no longer supported')

        logf, evalf = None, None
        if args.log_file:
            logf = open(args.log_file, 'a', encoding='utf-8')
            evalf = open(args.log_file + '.eval', 'a', encoding='utf-8')

        epoch = 0
        batch_nr = 0
        sent_nr = 0

        start_time = time()
        end_time = start_time + 3600*args.training_time
        best_test_xent = float('inf')

        # weight for the variable minibatch budget
        # FIXME: these need to be properly tuned
        # FIXME TOO: these are not used anymore, but perhaps should be (then
        #             in largetext.py)
        const_weight = 110
        src_weight = 1
        tgt_weight = 1
        x_weight = .045
        c_weight = .01
        pair_length = combo_len(0, tgt_weight, x_weight)

        def validate(test_pairs, start_time, optimizer, logf, sent_nr):
            result = 0.
            att_result = 0.
            t0 = time()
            for batch_pairs in iterate_batches(
                    test_pairs, config['batch_size']):
                test_x, test_y = prepare_batch(batch_pairs)
                test_outputs, test_outputs_mask, test_attention = test_y
                test_xent, test_xent_attention = model.xent_fun(
                        *(test_x + test_y))
                scale = (test_outputs.shape[1] /
                            (test_outputs_mask.sum()*np.log(2)))
                result += test_xent * scale
                att_result += test_xent_attention*scale
            print('%d\t%.3f\t%.3f\t%.3f\t%d\t%d' % (
                    int(t0 - start_time),
                    result,
                    att_result,
                    time() - t0,
                    optimizer.n_updates,
                    sent_nr),
                file=logf, flush=True)
            return result

        # only translate one minibatch for monitoring
        translate_src = test_src[:config['batch_size']]
        translate_trg = test_trg_unencoded
        #translate_trg = test_trg[:config['batch_size']]

        chrf_max = 0.0
        bleu_max = 0.0
        while time() < end_time:
            # Sort by combined sequence length when grouping training instances
            # into batches.
            for train_sent_pairs in train_iter:
                if logf and batch_nr % config['test_every'] == 0:
                    validate(test_pairs, start_time, optimizer, logf, sent_nr)

                print('Training batch size: %d' % len(train_sent_pairs),
                      flush=True)

                batch_src = [config['src_encoder'].encode_sequence(src_sent)
                             for src_sent, trg_sent in train_sent_pairs]
                batch_trg = [config['trg_encoder'].encode_sequence(trg_sent)
                             for src_sent, trg_sent in train_sent_pairs]
                batch_links_maps = [(None, None, None)]*len(batch_src)
                batch_pairs = list(zip(batch_src, batch_trg, batch_links_maps))

                sent_nr += len(batch_pairs)

                x, y = prepare_batch(batch_pairs)

                # This code can be used to print parameter and gradient
                # statistics after each update, which can be useful to
                # diagnose stability problems.
                #grads = [np.asarray(g) for g in optimizer.grad_fun()(*(x + y))]
                #print('Parameter summary:')
                #model.summarize(grads)
                #print('-'*72, flush=True)

                t0 = time()
                train_loss = optimizer.step(*(x + y))
                train_loss *= (y[0].shape[1] / (y[1].sum()*np.log(2)))
                print('Batch %d:%d of shape %s has loss %.3f (%.2f s)' % (
                    epoch+1, optimizer.n_updates,
                    ' '.join(str(m.shape) for m in (x[0], y[0])),
                    train_loss, time()-t0),
                    flush=True)
                if np.isnan(train_loss):
                    print('NaN loss, aborting!')
                    sys.exit(1)

                batch_nr += 1

                if config['save_every'] > 0 and batch_nr % config['save_every'] == 0:
                    filename = '%s.%d' % (args.save_model, optimizer.n_updates)
                    print('Writing model to %s...' % filename, flush=True)
                    with open(filename, 'wb') as f:
                        pickle.dump(config, f)
                        model.save(f)
                        optimizer.save(f)

                if batch_nr % config['translate_every'] == 0:
                    t0 = time()
                    test_dec = list(translate(translate_src, encode=False))
                    print('test_dec before', test_dec)
                    for src, trg_raw, trg_dec in zip(
                            translate_src, translate_trg, test_dec):
                        print('   SOURCE / TARGET / OUTPUT')
                        print(detokenize(
                            config['src_encoder'].decode_sentence(src),
                            config['source_tokenizer']))
                        print(detokenize(trg_raw, config['target_tokenizer']))
                        print(trg_dec)
                        print('-'*72)
                    print('Translation finished: %.2f s' % (time()-t0),
                          flush=True)

                    if config['target_tokenizer'] == 'char':
                        system = [detokenize(wordpunct_tokenize(s),'space')
                                  for s in test_dec]
                        reference = [
                            detokenize(wordpunct_tokenize(
                                detokenize(s, 'char')),
                                'space')
                            for s in translate_trg]
                    else:
                        reference = [detokenize(
                            s, config['target_tokenizer'])
                                     for s in translate_trg]
                        system = test_dec

                    bleu_result = BLEU(system,[reference])
                    chrf_result = chrF(reference,system)
                    is_best = chrf_result[0] >= chrf_max
                    chrf_max = max(chrf_result[0], chrf_max)
                    bleu_max = max(bleu_result[0], bleu_max)
                    print('BLEU = %f (%f, %f, %f, %f, BP = %f)' %
                            bleu_result)
                    print('chrF = %f (precision = %f, recall = %f)' %
                            chrf_result)

                    if evalf:
                        print('%d\t%.3f\t%.3f\t%d\t%d' % (
                                int(t0 - start_time),
                                bleu_result[0],
                                chrf_result[0],
                                optimizer.n_updates,
                                sent_nr),
                            file=evalf, flush=True)

                    if is_best:
                        with open(args.save_model, 'wb') as f:
                            pickle.dump(config, f)
                            model.save(f)
                            optimizer.save(f)


                #print('lambda_a = %g' % model.lambda_a.get_value())
                model.lambda_a.set_value(np.array(
                    model.lambda_a.get_value() * config['alignment_decay'],
                    dtype=theano.config.floatX))
                if time() >= end_time: break

            epoch += 1

        if logf: logf.close()
        if evalf: evalf.close()
        print('Training finished, saving final model', flush=True)

        with open(args.save_model + '.final', 'wb') as f:
            pickle.dump(config, f)
            model.save(f)
            optimizer.save(f)


if __name__ == '__main__': main()

