"""Text processing.

The :class:`TextEncoder` class is the main feature of this module, the helper
functions were used in earlier examples and should be phased out.
"""

from collections import Counter, namedtuple

import numpy as np
import theano

Encoded = namedtuple('Encoded', ['sequence', 'unknown'])

class TextEncoder(object):
    def __init__(self,
                 max_vocab=None,
                 min_count=None,
                 vocab=None,
                 counts=None,
                 sequences=None,
                 sub_encoder=None,
                 special=('<S>', '</S>', '<UNK>')):
        self.sub_encoder = sub_encoder
        self.special = special

        if vocab is not None:
            self.vocab = vocab
        else:
            if sequences is not None or counts is not None:
                c = counts if counts else \
                        Counter(x for xs in sequences for x in xs)
                if max_vocab is not None:
                    self.vocab = special + tuple(
                            s for s,_ in c.most_common(max_vocab))
                elif min_count is not None:
                    self.vocab = special + tuple(
                            s for s,n in c.items() if n >= min_count)
                else:
                    self.vocab = special + tuple(c.keys())

        self.index = {s:i for i,s in enumerate(self.vocab)}

    def __str__(self):
        if self.sub_encoder is None:
            return 'TextEncoder(%d)' % len(self)
        else:
            return 'TextEncoder(%d, %s)' % (len(self), str(self.sub_encoder))

    def __repr__(self):
        return str(self)

    def __getitem__(self, x):
        return self.index.get(x, self.index.get('<UNK>'))

    def __len__(self):
        return len(self.vocab)

    def encode_sequence(self, sequence, max_length=None, dtype=np.int32):
        """
        returns:
            an Encoded namedtuple, with the following fields:
            sequence --
                numpy array of symbol indices.
                Negative values index into the unknowns list,
                while positive values index into the encoder lexicon.
            unknowns --
                list of unknown tokens as Encoded(seq, None) tuples,
                or None if no subencoder.
        """
        start = (self.index['<S>'],) if '<S>' in self.index else ()
        stop = (self.index['</S>'],) if '</S>' in self.index else ()
        unk = self.index.get('<UNK>')
        unknowns = None if self.sub_encoder is None else []
        def encode_item(x):
            idx = self.index.get(x)
            if idx is None:
                if unknowns is None:
                    # NOTE: unk can be None if a word contains character
                    #       we have not seen before and the character
                    #       vocabulary was created without an <UNK> token
                    #       This workaround should not be necessary with new
                    #       vocabularies.
                    if unk is None: return 0
                    return unk
                else:
                    encoded_unk = self.sub_encoder.encode_sequence(x)
                    unknowns.append(encoded_unk)
                    return -len(unknowns)
            else:
                return idx
        encoded = tuple(idx for idx in list(map(encode_item, sequence))
                        if idx is not None)
        if max_length is None \
        or len(encoded)+len(start)+len(stop) <= max_length:
            out = start + encoded + stop
        else:
            out = start + encoded[:max_length-(len(start)+len(stop))] + stop
        return Encoded(np.asarray(out, dtype=dtype), unknowns)

    def decode_sentence(self, encoded):
        start = self.index.get('<S>')
        stop = self.index.get('</S>')
        return [''.join(self.sub_encoder.decode_sentence(
                    encoded.unknown[-x-1]))
                if x < 0 else self.vocab[x]
                for x in encoded.sequence
                if x not in (start, stop)]

    def pad_sequences(self, encoded_sequences,
                      max_length=None, pad_right=True,
                      fake_hybrid=False,
                      dtype=np.int32):
        """
        arguments:
            encoded_sequences -- a list of Encoded(encoded, unknowns) tuples.
            fake_hybrid -- if True, create a dummy unknown word matrix
                               (use if there is no subencoder)
        """
        if not encoded_sequences:
            # An empty matrix would mess up things, so create a dummy 1x1
            # matrix with an empty mask in case the sequence list is empty.
            m = np.zeros((1 if max_length is None else max_length, 1),
                         dtype=dtype)
            mask = np.zeros_like(m, dtype=np.bool)
            return m, mask

        length = max((len(x[0]) for x in encoded_sequences))
        length = length if max_length is None else min(length, max_length)

        m = np.zeros((length, len(encoded_sequences)), dtype=dtype)
        mask = np.zeros_like(m, dtype=np.bool)

        all_unknowns = []
        for i,pair in enumerate(encoded_sequences):
            encoded, unknowns = pair
            if unknowns is not None:
                unk_offset = len(all_unknowns)
                encoded = [idx - unk_offset if idx < 0 else idx
                           for idx in encoded]
                all_unknowns.extend(unknowns)

            if pad_right:
                m[:len(encoded),i] = encoded
                mask[:len(encoded),i] = 1
            else:
                m[-len(encoded):,i] = encoded
                mask[-len(encoded):,i] = 1

        if self.sub_encoder is None and not fake_hybrid:
            return m, mask
        else:
            if self.sub_encoder is None and fake_hybrid:
                char = np.zeros((1, 1), dtype=dtype)
                char_mask = np.zeros_like(char, dtype=np.bool)
            else:
                char, char_mask = self.sub_encoder.pad_sequences(all_unknowns)
            return m, mask, char, char_mask

    def decode_padded(self, m, mask, char=None, char_mask=None):
        if char is not None:
            unknowns = list(map(
                ''.join, self.sub_encoder.decode_padded(char, char_mask)))
        start = self.index.get('<S>')
        stop = self.index.get('</S>')
        return [[unknowns[-x-1] if x < 0 else self.vocab[x]
                 for x,b in zip(row,row_mask)
                 if bool(b) and x not in (start, stop)]
                for row,row_mask in zip(m.T,mask.T)]

