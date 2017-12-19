#!/usr/bin/env python3

# Script to create a TextEncoder object for storing the vocabulary

import sys
import argparse
import pickle
from collections import Counter

from text import TextEncoder


def main():
    parser = argparse.ArgumentParser(
            description='Vocabulary creation tool')
    parser.add_argument('--hybrid', action='store_true',
            help='Create a hybrid word/character vocabulary')
    parser.add_argument('--vocabulary', type=int, default=50000,
            help='Size of word vocabulary')
    parser.add_argument('--char-vocabulary', type=int, default=200,
            help='Maximum size of character vocabulary')
    parser.add_argument('--lowercase', action='store_true',
            help='Lower-case all data')
    parser.add_argument('--output', type=str, metavar='FILE', required=True,
            help='File to save the pickled TextEncoder object to')
    parser.add_argument('--min-char-count', type=int, default=1,
            help='Minimum character count to include (the default value of 1 '
                 'means that all characters are included)')
    parser.add_argument('--tokenizer', type=str, default='space',
            help='One of "space", "char" or "word"')
    parser.add_argument('files', metavar='FILE', type=str, nargs='+',
            help='Text file(s) to process')

    args = parser.parse_args()

    if args.tokenizer == 'char':
        tokenize = lambda s: list(s.strip())
    elif args.tokenizer == 'space' or args.tokenizer == 'bpe':
        tokenize = str.split
    elif args.tokenizer == 'word':
        import nltk
        from nltk import word_tokenize as tokenize

    token_count = Counter()
    char_count = Counter()

    character = args.tokenizer == 'char'

    # This actually makes sense as a workaround for character-based encoders
    #assert not (character and args.hybrid)

    for filename in args.files:
        print('Processing %s...' % filename, file=sys.stderr, flush=True)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.lower() if args.lowercase else line
                tokens = tokenize(line)
                token_count.update(tokens)
                if args.hybrid:
                    char_count.update(''.join(tokens))

    print('Creating encoder...', file=sys.stderr, flush=True)
    if args.hybrid:
        char_encoder = TextEncoder(
                counts=char_count,
                min_count=args.min_char_count,
                max_vocab=args.char_vocabulary,
                special=('<UNK>',))
        encoder = TextEncoder(
                counts=token_count,
                max_vocab=args.vocabulary,
                sub_encoder=char_encoder)
    else:
        encoder = TextEncoder(
                counts=token_count,
                max_vocab=args.vocabulary,
                min_count=args.min_char_count if character else None,
                special=('<S>', '</S>') + (() if character else ('<UNK>',)))

    print('Writing to %s...' % args.output, file=sys.stderr, flush=True)
    with open(args.output, 'wb') as f:
        pickle.dump(encoder, f, -1)

if __name__ == '__main__': main()

