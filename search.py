"""Search algorithms for recurrent networks."""

from collections import namedtuple

import theano
import numpy as np

def beam_with_coverage(
        step,
        states0,
        batch_size,
        start_symbol,
        stop_symbol,
        max_length,
        beam_size=8,
        min_length=0):
    """Beam search algorithm.

    See the documentation for :meth:`greedy()`.
    The only additional argument for this method is the opitonal `beam_size`.

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

    n_states = len(states0)

    # (beam, position, batch)
    sequence = np.full((1, 1, batch_size), start_symbol, dtype=np.int64)
    # (beam, position, batch)
    mask = np.ones((1, 1, batch_size), dtype=theano.config.floatX)
    # (beam, batch, dims)
    states = [s[None,:,:] for s in states0]
    # (beam, batch)
    scores = np.zeros((1, batch_size))
    # (beam, batch, source_position)
    coverage = np.zeros((1, batch_size, FIXME))

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

        # (beam_size, batch_size)
        best = np.argsort(all_dists.T, axis=0)[-beam_size:, :]
        # (beam_size, batch_size)
        best_beam = np.floor_divide(best, n_symbols)
        best_symbol = best - (n_symbols*best_beam)

        # TODO: optimize by allocating sequence/mask in the beginning,
        #       then shrink if necessary before returning
        sequence = np.concatenate([
                np.swapaxes(
                    sequence[best_beam,:,np.arange(batch_size)[None,:]],
                    1, 2),
                best_symbol[:,None,:]], axis=1)
        last_active = (sequence[:,-2,:] != stop_symbol)
        mask = np.concatenate([
                np.swapaxes(
                    mask[best_beam,:,np.arange(batch_size)[None,:]],
                    1, 2),
                last_active[:,None,:]], axis=1)
        states = [s[best_beam,np.arange(batch_size)[None,:],:]
                  for s in all_states]
        scores = all_dists.T[best,np.arange(batch_size)[None,:]]

        if not mask[:,-1,:].any():
            return sequence, mask, scores

    n_beams = sequence.shape[0]
    sequence = np.concatenate(
            [sequence,
             np.full((n_beams, 1, batch_size), stop_symbol, dtype=np.int64)],
            axis=1)
    mask = np.concatenate([mask, mask[:,-1:,:]], axis=1)

    return sequence, mask, scores

