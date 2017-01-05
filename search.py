"""Search algorithms for recurrent networks."""

from collections import namedtuple

import itertools
import numpy as np
import theano

Hypothesis = namedtuple(
    'Hypothesis',
    ['sentence',    # index of sentence in minibatch
     'score',       # raw or adjusted score
     'history',     # sequence up to last symbol
     'last_sym',    # last symbol
     'state',       # RNN state
     'coverage']    # accumulated coverage

def by_sentence(beams):
    key = lambda hyp: hyp.sentence
    return itertools.groupby(sorted(beams, key=key), key)

def beam_with_coverage(
        step,
        states0,
        batch_size,
        start_symbol,
        stop_symbol,
        max_length,
        beam_size=8,
        min_length=0,
        alpha=0.2,
        beta=0.2,
        prune_margin=3.0):
    """Beam search algorithm.

    See the documentation for :meth:`greedy()`.
    The additional arguments are FIXME
    prune_margin is misleadingly named beamsize in Wu et al 2016

    Returns
    -------
    outputs : numpy.ndarray(int64)
        Array of shape ``(n_beams, length, batch_size)`` with the output
        sequences. `n_beams` is less than or equal to `beam_size`.
    outputs_mask : numpy.ndarray(theano.config.floatX)
        Array of shape ``(n_beams, length, batch_size)``, containing the
        mask for `outputs`.
    scores : numpy.ndarray(float64)
        Array of shape ``(n_beams, batch_size)``.
        Log-probability of the sequences in `outputs`.
    """

    beams = [Hypothesis(i, 0., (), start_symbol,
                        [s[i, :] for s in states0],
                        0.)
             for i in range(batch_size)]

    n_states = len(states0)

    for i in range(max_length-2):
        # build step inputs
        active = [hyp for hyp in beams if hyp.last_sym != stop_symbol]
        completed = [hyp for hyp in beams if hyp.last_sym == stop_symbol]
        if len(active) == 0:
            return by_sentence(beams)

        states = []
        prev_syms = np.zeros((1, len(active)), dtype=np.int64)
        mask = np.ones((len(active),), dtype=theano.config.floatX)
        for (j, hyp) in enumerate(active):
            states.append(hyp.states)
            prev_syms[0, j] = hyp.last_sym
        states = [np.array(x) for x in zip(*states)]

        # predict
        all_states, all_dists, attention = step(i, states, prev_syms, mask)
        if i <= min_length:
            all_dists[:, stop_symbol] = 1e-30
        n_symbols = all_dists.shape[-1]
        # (target_pos, batch, source_pos) -> (batch, source_pos)
        attention = attention.sum(axis=0)

        # extend active hypotheses
        extended = []
        for (j, hyp) in enumerate(active):
            history = hyp.history + hyp.last_sym
            for symbol in range(n_symbols):
                score = hyp.score + all_dists[j, symbol]
                coverage = hyp.coverage + attention[j, :]
                if symbol == stop_symbol:
                    # FIXME: preprune to avoid normalizing unnecessarily
                    pass
                extended.append(
                    Hypothesis(hyp.sentence,
                               score,
                               history,
                               symbol,
                               [s[j, :] for s in all_states],
                               coverage)

        # prune
        beams = []
        for (_, group) in by_sentence(completed + extended):
            beams.extend(sorted(group, key=lambda hyp: -hyp.score)[:beam_size])
    return by_sentence(beams)
