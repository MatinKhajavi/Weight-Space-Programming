Most of the remaining loops are now of a different kind. Original loops were 88% verification, 4% wrap-up and 5% other. Under E3 the split is 42% verification, 13% wrap-up and 41% "other", which match neither pattern. (The last part of the run was a bookkeeping error in my own indexing code and does not affect these counts.) The "other" group is now the largest single bucket, and I need to see what it repeats. It could be re-reading the problem, restating a formula or listing cases, and each would point to a different mechanism. I will print short excerpts of several "other" loops, plus their line openers.

```python
# Inspect 'other' category of remaining E3 loops (short excerpts), with their openers
oth = [(r['key'], int(r['P'])) for _, r in LS11[LS11.loop].iterrows() if categorize(G11[r['key']]['text'], int(r['P']))=='other']
rng3 = np.random.RandomState(5)
for i_ in rng3.choice(len(oth), 8, replace=False):
    k_, P_ = oth[i_]; per = G11[k_]['text'][-P_:]
    print(k_[0][:26], 'P', P_, '| openers', loop_opener(G11[k_]['text'], P_, len(G11[k_]['text'])-3*P_)[:5], '|', repr(per[:200]))
```