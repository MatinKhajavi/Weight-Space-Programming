```python
# Hypothesis: inspect the two early-onset loop samples to judge suitability for the attention deep-dive.
for i in [417, 437]:
    s_ = samples[i]; ids = toks(s_['text']); r = dl[dl.i==i].iloc[0]
    print('=====', s_['id'], 'p', r.p, 'onset', r.onset, 'prompt toks', len(toks(prompt_text(s_['id']))), 'ref', s_['ref'])
    print('PROBLEM:', prob_by_id[s_['id']]['problem'][:300])
    print('PRE-ONSET tail:', repr(tok.decode(ids[r.onset-200:r.onset])))
    print('CYCLE:', repr(tok.decode(ids[r.onset:r.onset+r.p])))
```