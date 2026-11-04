```python
# Hypothesis: loop cycles are dominated by "Wait"-style re-verification; check cycle content keywords.
kw = ['Wait', 'answer', 'Answer', 'ready', 'write', 'response', 'check', 'Check', 'correct', 'Yes', 'boxed', 'Final', 'So ', 'Is ', '?']
cnt = {k:0 for k in kw}
cyc_texts = []
for r in dl.itertuples():
    s_ = samples[r.i]; ids = toks(s_['text'])
    cyc = tok.decode(ids[-r.p:])  # one full period at the end
    cyc_texts.append(cyc)
    for k in kw:
        if k in cyc: cnt[k] += 1
print(len(cyc_texts), {k: round(v/len(cyc_texts),2) for k,v in cnt.items()})
for c in cyc_texts[10:16]:
    print(repr(c[:250]))
```