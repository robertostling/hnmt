import sys
import re

import unicodedata
from hanziconv import HanziConv

RE_LATIN_SPACE = re.compile(
    r'(?<=[a-zA-Z\u00c0-\u01ff])\s+(?=[a-zA-Z\u00c0-\u01ff])')
RE_SPACES = re.compile(r'\s+')

def process(line):
    line = line.strip()
    line = RE_LATIN_SPACE.sub('‧', line)
    line = RE_SPACES.sub('', line)
    line = unicodedata.normalize('NFKC', line)
    line = HanziConv.toSimplified(line)
    return line

if __name__ == '__main__':
    for line in sys.stdin:
        print(process(line))

