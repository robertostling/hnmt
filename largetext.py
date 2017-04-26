from random import shuffle, Random
from operator import itemgetter

import numpy as np

class ShuffledTextIterator:
    """Iterator class used internally by ShuffledText"""
    def __init__(self, text):
        self.offsets = list(text.offsets)
        self.random = text.random
        shuffle(self.offsets, self.random.random)
        self.f = text.f
        self.buf = []
        self.buf_size = text.block_size*text.max_blocks
        self.block_size = text.block_size
        self.buf_used = 0
        self.encoding = text.encoding

    def _fill_buffer(self):
        modified = False
        while self.offsets and self.buf_used < self.buf_size:
            start = self.offsets.pop()
            stop = start + self.block_size
            self.f.seek(start)
            # We might be in the middle of a line, so throw away the first
            # one unless we're at the beginning of the file
            if start > 0:
                line = self.f.readline()
            while line and self.f.tell() <= stop:
                line = self.f.readline()
                if line:
                    self.buf.append(line.rstrip(b'\n'))
                    modified = True
        if modified:
            shuffle(self.buf, self.random.random)

    def __iter__(self):
        return self

    def __next__(self):
        self._fill_buffer()
        if self.buf:
            return str(self.buf.pop(), self.encoding)
        else:
            raise StopIteration


class ShuffledText:
    """Iterate through the lines of a large text file in pseudo-random order.

    This class will keep an internal buffer of lines using approximately
    max_blocks*block_size bytes, which is refilled by seeking into a random
    block of the file and shuffling the lines.

    Note that the input file f must be opened in binary mode, while the
    iterator returns str objects with an encoding specified by the encoding
    argument.

    Randomization is done once (in __iter__) for the block order, then the
    buffer is regularly shuffled. This two-step process means that the
    distance between two lines in the resulting iterator has a weak
    correlation (inversely proportional to max_blocks) with the distance in
    the input file.

    If this becomes a problem, you may want to increase max_blocks (and
    possible decrease block_size if RAM usage matters) in order to reduce the
    bias in the final output stream.

    By default the RAM buffer is about 128 MB, and the ratio between
    block_size and max_blocks is chosen to provide reasonable performance
    even with mechanical hard disks.

    f -- file to iterate through, this is normally a Python file object of
         a large file opened in binary reading mode ('rb'). It must support
         seeking.
    block_size -- each block of lines is approximately this many bytes
    max_blocks -- size of the internal buffer, in blocks
    encoding -- text encoding
    """
    def __init__(self, f, block_size=0x40000, max_blocks=512,
                 encoding='utf-8', seed=123):
        assert 'b' in f.mode
        self.random = Random(seed)
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.lines = None
        self.f = f
        self.encoding = encoding

        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)

        if file_size <= block_size*max_blocks:
            # For sufficiently small files, read the whole one into RAM
            self.lines = [str(line.rstrip(b'\n'), encoding) for line in f]
            shuffle(self.lines, self.random.random)
        else:
            # For large files, only store a list of block offsets
            self.offsets = list(range(0, file_size, block_size))

    def __iter__(self):
        if self.lines: return iter(self.lines)
        self.f.seek(0, 0)
        return ShuffledTextIterator(self)


class HalfSortedIterator:
    """Iterate over minibatches in semi-sorted order.

    This class iterates over lists of items produced by mapping the input
    iterator (lines) through the function preprocess.

    Note that order within a minibatch is considered irrelevant, and in this
    implementation a zigzag pattern is returned.

    By specifying a number for max_items, the batch size in items is limited.
    By specifying a number for max_area, the maximum size (rows * cols) for
    the resulting minibatch matrix is limited.

    preprocess -- function from str to whatever object the batches consist of
                  that does any preprocessing (e.g. tokenization, indexing)
    length -- function returning the length of a batch item (i.e. what is
              returned by the preprocess function
    max_items -- if max_area is None, the batch size constraint is that
                 batches may contain at most this many items.
                 if max_area is None, this argument is ignored
    max_area -- if not None, the batch size constraint is that its largest
                element (as given by the length function) times the batch size
                must be at most max_area
    n_blocks -- the input will be sorted in blocks of approximately this many
                batches
    """
    def __init__(self, lines, preprocess=lambda x: x, length=len,
                 max_items=64, max_area=None, n_blocks=16):
        # TODO: consider more flexible specification of length constraint,
        # e.g. to support attention vectors
        self.lines = lines
        self.preprocess = preprocess
        self.length = length
        self.max_items = max_items
        self.max_area = max_area
        self.n_blocks = n_blocks
        self.eof = False
        self.reverse = False
        self.buf = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.eof: raise StopIteration

        batch = []
        largest = 0
        while True:
            if not self.buf:
                self._fill_buf()
                if not self.buf:
                    self.eof = True
                    if batch: return batch
                    raise StopIteration

            tos = self.buf[-1]
            if self.max_area:
                size = max(largest, self.length(tos)) * (len(batch)+1)
                full = size > self.max_area
            else:
                size = len(batch)+1
                full = size > self.max_items

            if full:
                return batch

            batch.append(self.buf.pop())
            largest = max(largest, self.length(tos))

    def _fill_buf(self):
        assert not self.buf

        size = 0
        max_size = self.n_blocks * \
                   (self.max_area if self.max_area else self.max_items)
        while size < max_size:
            try:
                line = next(self.lines)
            except StopIteration:
                break
            item = self.preprocess(line)
            self.buf.append(item)
            size += self.length(item) if self.max_area else 1

        self.buf.sort(key=self.length, reverse=self.reverse)
        self.reverse = not self.reverse


