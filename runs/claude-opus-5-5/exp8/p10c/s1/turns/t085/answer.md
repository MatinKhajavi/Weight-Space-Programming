Most remaining v2 loops sit inside the model's nested bullet lists, so my first-word count mostly caught the bullet marker "*". The visible words still show a shift: "So" now starts lines in 20% of loops, "Wait" in 17% (39% in the original) and "But" in 9%.

The excerpts look like conclude→doubt→re-check cycles over the same point, for example:
- "So the order is 4." → "Wait, I need to check if f(x) is defined as …" (repeated)
- "Let's check if there's a misunderstanding of …"

The reference answer was already written before loop onset in 80 of the 144 v2 loops. So many loops are "solved but keeps re-checking".

Now I will look past the bullet markers to the real first word of each cycle line. I will measure how many v2 cycles contain any doubt or re-check phrase versus being pure computation or restatement. I will also check that the fwd_4 activation data (original model, two natural loop texts, all layers) is still loaded for a new attribution analysis. If most v2 cycles still carry doubt or re-check phrases, the next target is the remaining doubt pathway upstream of the ablated neuron 998 in layer 16. If most are pure recomputation, the copy circuit is the better target.

```python
# Hypothesis: remaining v2 loop cycles are mostly doubt/re-check cycles (not pure recomputation); count first words after bullet markers and doubt-phrase content
doubt_re = re.compile(r"\b(Wait|wait|But|Actually|Hmm|Hold|Maybe|maybe|double-check|double check|re-read|reread|misunderstand|misinterpret|check if|trick|Is it possible|Let's check|verify|Let's re)\b")
def fw2(c2):
    out = set()
    for l in c2.split('\n')[1:-1]:
        l2 = re.sub(r"^\s*(\*+|-|\d+\.|>)\s*", "", l).strip()
        if not l2: continue
        m = re.match(r"([A-Za-z']+|\S)", l2); out.add(m.group(1) if m else '?')
    return out
for k,res in C.items():
    cnt = collections.Counter(); nd = 0; nmath = 0
    for d in res:
        G = {'orig':g9o,'v1':g8e,'v2':g9e}[k]
        d['fw2'] = fw2(d['c'] + d['c'])
        cnt.update(d['fw2'])
        d['doubt'] = bool(doubt_re.search(d['c']))
        nd += d['doubt']
    n = len(res)
    print(f"{k}: loops {n}, cycles with doubt/re-check phrase {nd} ({nd/n:.2f})")
    print('   first words after bullets:', ', '.join(f"{w}:{c/n:.2f}" for w,c in cnt.most_common(20) if w.isascii()))
import glob
print('fwd_4 files:', sorted(os.path.basename(x) for x in glob.glob('/work/fwd_4/*'))[:40])
print('kernel arrays:', [k for k,v in globals().items() if isinstance(v, np.ndarray) and v.size>1000][:30])
```