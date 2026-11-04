`analyze` gives per-sample loop onset, period, doubt-word counts and whether each doubt word sits inside the cycle. The cut sample I tried was one of the few non-exact cases.

Next I will run a census of loop cycles across the original, v1 and v2 runs on the same 256 setA problems. For each exact loop I will measure:
- which words start the lines inside the cycle;
- whether the cycle contains a doubt word, the reference answer or a \boxed{};
- whether the correct answer appeared before the loop began.

Ablating the "Wait" neuron moved doubts toward "But". So my hypothesis is that the remaining v2 loops are led by other doubt words, or are doubt-free restatements of an already found answer, rather than Wait-led. A table where v2 cycles start mainly with a few non-Wait words would point to new neuron-level targets. Mostly verbatim recomputation or restatement would point back at the copy circuit. I will also print a few short cycle excerpts to judge this directly.

```python
# Hypothesis: remaining v2 loops are not Wait-led; characterize cycle content (line-start words, doubt words, answer presence) for orig / v1 / v2 on setA
import collections, re, random
g8e = readjl('/work/gen_8_edited.jsonl'); g9o = readjl('/work/gen_9_original.jsonl')
def census(G):
    res = []
    for rr in G:
        s = rr['samples'][0]
        if not s['cut']: continue
        ids = toks(s['text']); on, p = onset_period(ids)
        if p is None: continue
        s2 = tok.decode(ids[-2*p:]); parts = s2.split('\n')
        lines = set(l.strip() for l in parts[1:-1] if l.strip())
        fw = set()
        for l in lines:
            m = re.match(r"([A-Za-z']+|\S)", l); fw.add(m.group(1) if m else '?')
        c = tok.decode(ids[-p:])
        res.append(dict(id=rr['id'], on=on, p=p, c=c, fw=fw, nl=len(parts)-2, box=('boxed' in c),
                        ansin=(norm_ans(rr['answer']) in c.replace(' ','')) if rr['answer'] else False,
                        before=ans_before(ids, on, rr['answer'])))
    return res
C = {k: census(G) for k,G in [('orig',g9o),('v1',g8e),('v2',g9e)]}
for k,res in C.items():
    cnt = collections.Counter(w for d in res for w in d['fw'])
    n = len(res)
    print(f"{k}: loops {n} | median p {np.median([d['p'] for d in res]):.0f} | onset med {np.median([d['on'] for d in res]):.0f} | no-newline cycles {sum(d['nl']<1 for d in res)} | boxed in cycle {sum(d['box'] for d in res)} | ref ans in cycle {sum(d['ansin'] for d in res)} | ans before onset {sum(bool(d['before']) for d in res)}")
    print('   top line-start words (share of loops):', ', '.join(f"{w}:{c/n:.2f}" for w,c in cnt.most_common(22) if w.isascii()))
random.seed(0)
for d in random.sample(C['v2'], 6):
    print('--', d['id'], 'p', d['p'], 'on', d['on'], '|', d['c'][:260].replace('\n',' | '))
```