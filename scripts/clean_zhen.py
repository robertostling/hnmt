import sys
from multiprocessing import Pool

import langid
#from langdetect import detect

def count_hanzi(s):
    return sum(ord(c) >= 0x4e00 and ord(c) <= 0x9fff for c in s)

def clean_pair(pair):
    bytes_en, bytes_zh = pair
    line_en = str(bytes_en, 'utf-8')
    line_zh = str(bytes_zh, 'utf-8')
    en_words_list = line_en.split()
    en_words = len(en_words_list)
    if en_words < 2: return None
    if en_words > 130 or len(line_en) > 1000: return None
    zh_chars = count_hanzi(line_zh)
    if zh_chars < 2: return None
    if zh_chars > 200 or len(line_zh) > 300: return None
    ratio = zh_chars / en_words
    # Empericially determined from the 1st and 99th percentile in a sample of
    # the UN corpus
    if ratio < 0.6 or ratio > 2.6: return None
    #if not detect(line_en) == 'en': return None
    #if not detect(line_zh)[:2] == 'zh': return None
    if not langid.classify(line_en)[0] == 'en': return None
    if not langid.classify(line_zh)[0] == 'zh': return None
    return ' '.join(en_words_list), ' '.join(line_zh.split())

def clean_multi(inf_en, inf_zh, outf_en, outf_zh):
    with Pool() as pool:
        for en_zh in pool.imap(clean_pair, zip(inf_en, inf_zh), 1000):
            if en_zh is not None:
                en, zh = en_zh
                print(en, file=outf_en)
                print(zh, file=outf_zh)

def clean_single(inf_en, inf_zh, outf_en, outf_zh):
    for en, zh in zip(inf_en, inf_zh):
        en_zh = clean_pair((en, zh))
        if en_zh is None: continue
        en, zh = en_zh
        print(en, file=outf_en)
        print(zh, file=outf_zh)


if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as inf_en, \
         open(sys.argv[2], 'rb') as inf_zh, \
         open(sys.argv[1]+'.clean', 'w', encoding='utf-8') as outf_en, \
         open(sys.argv[2]+'.clean', 'w', encoding='utf-8') as outf_zh:
        #clean_multi(inf_en, inf_zh, outf_en, outf_zh)
        clean_single(inf_en, inf_zh, outf_en, outf_zh)

