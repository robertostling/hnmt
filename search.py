"""Search algorithms for recurrent networks."""

from collections import namedtuple

import theano
import numpy as np

Hypothesis = namedtuple(
    'Hypothesis',
    ['sentence',    # index of sentence in minibatch
     'score',       # raw or adjusted score
     'history',     # sequence up to last symbol
     'last_sym',    # last symbol
     'state',       # RNN state
     'coverage']    # accumulated coverage

def beam_with_coverage(
        step,
        states0,
        batch_size,
        start_symbol,
        stop_symbol,
        max_length,
        source_length,
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
                        np.zeros((source_length,)))
             for i in range(batch_size)]

    n_states = len(states0)

    # (beam, position, batch)
    #sequence = np.full((1, 1, batch_size), start_symbol, dtype=np.int64)
    # (beam, position, batch)
    #mask = np.ones((1, 1, batch_size), dtype=theano.config.floatX)
    # (beam, batch, dims)
    #states = [s[None,:,:] for s in states0]
    # (beam, batch)
    #scores = np.zeros((1, batch_size))
    # (beam, batch, source_position)
    #coverage = np.zeros((1, batch_size, source_length))

    for i in range(max_length-2):
        # build step inputs
        active = [hyp for hyp in beams if hyp.last_sym != stop_symbol]
        completed = [hyp for hyp in beams if hyp.last_sym == stop_symbol]

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
                if symbol == stop_symbol:
                    # FIXME: preprune to avoid normalizing unnecessarily
                    pass
                extended.append(
                    Hypothesis(hyp.sentence,
                               score,
                               history,
                               symbol,
                               [s[j, :] for s in all_states],
                               hyp.coverage + attention[j, :])

        # prune

    ### FIXME: below this old stuff

    for i in range(max_length-2):
        # Current size of beam
        n_beams = sequence.shape[0]

        all_states = []
        all_dists = []
        for j in range(n_beams):
            part_states, part_dists = step(
                i, [s[j,...] for s in states], sequence[j,...], mask[j,...])
            if i <= min_length:
                part_dists[:, stop_symbol] = 1e-30
            # Hard constraint: </S> must always be followed by </S>
            finished = (sequence[j, -1, :] == stop_symbol)[...,None]
            finished_dists = np.full_like(part_dists, 1e-30)
            finished_dists[:, stop_symbol] = 1.0
            part_dists = part_dists*(1-finished) + finished_dists*finished
            all_states.append(part_states)
            all_dists.append(part_dists)
            # FIXME: length and coverage penalties for newly finished beams
            # FIXME: only add penalty once, for first </S>. Keep track of length?

        # list of (n_beams, batch_size, dims)
        all_states = [np.array(x) for x in zip(*all_states)]
        # (n_beams, batch_size, n_symbols)
        all_dists = np.log(np.array(all_dists)) + scores[:,:,None]

        n_symbols = all_dists.shape[-1]

        # (batch_size, n_beams*n_symbols)
        all_dists = np.concatenate(list(all_dists), axis=-1)
        n_hyps = all_dists.shape[-1]

        # FIXME: new procedure
        idx_to_beam = np.floor_divide(np.arange(n_hyps), n_symbols)
        idx_to_sym = np.mod(np.arange(n_hyps), n_symbols)
        # find best score in all_dists
        # (batch_size,)
        best = np.max(all_dists.T, axis=0)
        # prune hyps with score > best + prune_margin
        # (n_beams*n_symbols, batch_size)
        keep = all_dists.T > best + prune_margin
        pruned_dists = (all_dists.T)[keep]
        idx_to_beam = FIXME
        # if any remaining are newly finished, apply penalties
        # sort by adjusted score, keep top

        # (beam_size, batch_size)
        top = np.argsort(all_dists.T, axis=0)[-beam_size:, :]
        # (beam_size, batch_size)
        top_beam = np.floor_divide(top, n_symbols)
        top_symbol = top - (n_symbols*top_beam)

        # TODO: optimize by allocating sequence/mask in the beginning,
        #       then shrink if necessary before returning
        sequence = np.concatenate([
                np.swapaxes(
                    sequence[top_beam,:,np.arange(batch_size)[None,:]],
                    1, 2),
                top_symbol[:,None,:]], axis=1)
        last_active = (sequence[:,-2,:] != stop_symbol)
        mask = np.concatenate([
                np.swapaxes(
                    mask[top_beam,:,np.arange(batch_size)[None,:]],
                    1, 2),
                last_active[:,None,:]], axis=1)
        states = [s[top_beam,np.arange(batch_size)[None,:],:]
                  for s in all_states]
        scores = all_dists.T[top,np.arange(batch_size)[None,:]]

        if not mask[:,-1,:].any():
            return sequence, mask, scores

    n_beams = sequence.shape[0]
    sequence = np.concatenate(
            [sequence,
             np.full((n_beams, 1, batch_size), stop_symbol, dtype=np.int64)],
            axis=1)
    mask = np.concatenate([mask, mask[:,-1:,:]], axis=1)

    return sequence, mask, scores

