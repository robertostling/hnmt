import sys

best = {}

for line in sys.stdin:
    idx, text, score = line.rstrip('\n').split(' ||| ')
    idx = int(idx)
    score = float(score)
    if idx not in best or score > best[idx][1]:
        best[idx] = (text, score)

assert set(best.keys()) == set(range(len(best)))

for idx, (text, score) in sorted(best.items()):
    print(text)

