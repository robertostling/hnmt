"""n-best reranking with HNMT

Usage:
    1) create an n-best list with --nbest-list N
    2) extract the middle column, and score it with --score scores.txt
       NOTE: use the --score-repeat-source N argument with hnmt.py to load
       each source sentence N times, so it is properly aligned with the target
       n-best list (where only the middle column should be used!)
    3) rerank.py target.nbest scores.txt >target.txt

It is also possible to provide multiple n-best lits + score files, in which
case the highest-scoring sentence among all the file pairs is chosen.
"""

import sys

def rerank(nbest_file, scores_file):
    with open(scores_file) as f:
        scores = list(map(float, f))

    best = {}

    with open(nbest_file, 'r', encoding='utf-8') as f:
        for other_score, line in zip(scores, f):
            idx, sent, score = line.rstrip('\n').split(' ||| ')
            idx = int(idx)
            score = float(score) + other_score
            if score > best.get(idx, (None, float('-inf')))[1]:
                best[idx] = (sent, score)

    return best

if __name__ == '__main__':
    assert len(sys.argv[1:]) % 2 == 0
    best = None
    for i in range(0, len(sys.argv)-1, 2):
        nbest_file, scores_file = sys.argv[1+i:1+i+2]
        scores = rerank(nbest_file, scores_file)
        if best is None:
            best = scores
        else:
            assert len(scores) == len(best)
            for idx, (sent, score) in scores.items():
                if best[idx][1] > score:
                    best[idx] = (sent, score)

    for _, (sent, score) in sorted(best.items()):
        print(sent)

