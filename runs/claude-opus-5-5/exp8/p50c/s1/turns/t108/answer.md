Relative to Edit A, Edit C gains +25.0 points (SE 2.8): 66 problems went from wrong to right and only 2 the other way. The gain holds for MATH (+19.5) and GSM8K (+30.5). Against the original model it is +47.7 points (SE 3.2), with 123 wrong-to-right and 1 right-to-wrong. Correct answers also got shorter: median 5.9k tokens versus 9.2k under Edit A. The warning sign is that "Wait" has almost disappeared, from about 20 per 1,000 tokens in the original and 12.7 under A to a median of 0.1. Doubling the brake does more than break spirals: it removes the model's main self-correction word. I need to check that the model still re-examines its work in other words rather than having stopped checking. Otherwise this would look more like a global "stop doubting" knob than a repair. I'll compare the rates of other self-correction markers ("Hmm", "Actually", "But", "Let me check/double-check", "However", "Alternatively") across the three models. I'll also check how the remaining 29% of cut answers fail (exact loops, and what their cycles start with) and whether the 28 wrong answers are short, premature commitments. That tells me whether a gentler dose (×1.5) should be the main candidate and what drives the remaining failures.

```python
# Characterize Edit C: replacement of 'Wait' by other self-check markers; remaining cuts (loops?) and wrong answers
markers = {'Wait': r'\bWait\b', 'Hmm': r'\bHmm\b', 'Actually': r'\bActually\b', 'But': r'\bBut\b', 'check': r'(?i)\b(double-?check|let me check|let\'s check|verify)\b', 'However': r'\bHowever\b', 'Alternatively': r'\bAlternatively\b'}
for nm_, g in (('orig', gO), ('A', gA), ('C', gC)):
    rates = {k: np.median([1000*len(re.findall(p, g[i]['text']))/max(1,g[i]['ntok']) for i in S1]) for k, p in markers.items()}
    print(nm_, ' '.join('%s %.1f' % (k, v) for k, v in rates.items()))
# remaining cuts under C: exact periodic tails?
def tail_period(t, maxp=3000):
    tail = t[-6000:]
    for p_ in range(20, maxp):
        if tail[-2*p_:] [:p_] == tail[-p_:] and tail[-3*p_:-2*p_] == tail[-p_:]: return p_
    return None
cutC = [i for i in S1 if gC[i]['cut']]
pers = [tail_period(gC[i]['text']) for i in cutC]
print('C cut n=%d, exact-loop tails %d, median period chars %s' % (len(cutC), sum(p is not None for p in pers), np.median([p for p in pers if p]) if any(pers) else None))
starts = []
for i, p_ in zip(cutC, pers):
    if p_: 
        unit = gC[i]['text'][-p_:]; starts.append(re.sub(r'\s+', ' ', unit).strip()[:25])
print('loop unit starts (first 12):', [s.encode('ascii','replace').decode() for s in starts[:12]])
wrC = [i for i in S1 if (not gC[i]['ok']) and not gC[i]['cut']]
print('C wrong n=%d median tok %d; of these A was: cut %d, wrong %d, right %d' % (len(wrC), np.median([gC[i]['ntok'] for i in wrC]), sum(gA[i]['cut'] for i in wrC), sum((not gA[i]['ok']) and not gA[i]['cut'] for i in wrC), sum(gA[i]['ok'] for i in wrC)))
```