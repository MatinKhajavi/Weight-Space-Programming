The presence-penalty drift hypothesis is not supported where I can measure it. At newline positions in the recorded windows the indentation token is chosen with near certainty: median probability about 1.0 and a margin of 8–12 logits over the next depth. The -1.5 penalty would flip the choice in essentially none of 366 positions, in the original, A or C. So deep nesting is most likely the model's own choice, sub-bullets for each new doubt, and a symptom of doubt spirals rather than a sampler artifact I could repair. I am setting it aside. The largest remaining failure is MATH answers being cut off (35%). Before choosing between loop-breaking and leaving it, I want to know whether these cut-offs are concentrated on hard problems (levels 4–5, likely genuine failed searches) or also hit easy ones (a pathology still worth fixing). I will tabulate C's outcomes by MATH level and subject.

```python
# Are C's MATH cut-offs concentrated at hard levels (failed search) or also at easy levels (residual pathology)?
mref = {}
for l in open('/practice/math_train.jsonl'):
    r = json.loads(l); mref[r['id']] = (r.get('level'), r.get('subject'))
lv = {}
for name, g, ids in [('S1', g13, S1), ('H1', g16e, H1)]:
    for i in ids:
        if i.startswith('gsm'): continue
        s = g[i]; L_ = str(mref[i][0]); st = 'ok' if str(s['correct']) == 'True' else ('cut' if str(s['cut']) == 'True' else 'wrong')
        lv.setdefault(L_, Counter())[st] += 1
for L_ in sorted(lv):
    c = lv[L_]; n = sum(c.values()); print('level', L_, 'n', n, {k: f'{c[k] / n:.0%}' for k in ['ok', 'cut', 'wrong']})
# also original on same items for reference
lo = {}
for name, g, ids in [('S1', gO, S1), ('H1', load_gen('/work/gen_16_original.jsonl'), H1)]:
    for i in ids:
        if i.startswith('gsm'): continue
        s = g[i]; L_ = str(mref[i][0]); lo.setdefault(L_, Counter())['ok' if str(s['correct']) == 'True' else ('cut' if str(s['cut']) == 'True' else 'wrong')] += 1
print('orig:', {L_: f"{lo[L_]['ok'] / sum(lo[L_].values()):.0%} ok / {lo[L_]['cut'] / sum(lo[L_].values()):.0%} cut" for L_ in sorted(lo)})
```