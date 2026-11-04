```python
# Hypothesis: loops start after the model has already reached an answer; check reference answer presence before onset, and view onset context.
import re
def norm_ans(a): return a.replace('\\dfrac','\\frac').replace(' ','')
rows = []
for i, li in info:
    if li is None: continue
    s_ = samples[i]; p, onset, n = li
    ids = toks(s_['text'])
    pre = tok.decode(ids[:onset])
    ref = s_['ref']
    present = (norm_ans(ref) in pre.replace(' ','')) if len(ref)>0 else False
    # boxed occurrences before onset
    nbox = pre.count('\\boxed')
    rows.append(dict(i=i, id=s_['id'], src=s_['source'], p=p, onset=onset, ref_in_pre=present, nbox_pre=nbox))
dl = pd.DataFrame(rows)
print(dl.groupby('src')[['ref_in_pre']].mean(), dl.groupby('src')['nbox_pre'].apply(lambda x:(x>0).mean()))
for j in [0, 1, 7]:
    r = dl.iloc[j]; s_ = samples[r.i]; ids = toks(s_['text'])
    print('=====', r.id, 'p', r.p, 'onset', r.onset, 'ref', s_['ref'])
    print(repr(tok.decode(ids[r.onset-120:r.onset])), '\n  >>> ', repr(tok.decode(ids[r.onset:r.onset+2*r.p])[:400]))
```