Under C, 127 answers were cut off (S1 and H1 combined). 80 end in exact loops: 46 multi-line reasoning cycles, 16 short in-line stutters, 12 'Maybe' lines and 6 repeated answer statements. The other 47 have no exact period. A striking feature of the loop units is very deep indentation (up to 80 spaces of nested bullets) with doubt lines like "*   Contradiction." / "*   Okay, maybe the roles are...". One plausible contributor comes from the fixed protocol itself: the presence penalty only penalizes tokens already used, so fresh, deeper indentation tokens are favoured over used ones. That would ratchet long texts into ever deeper nesting. Before choosing a target, I need to see the 47 non-loop cuts: genuine long exploration, near-loops such as endless enumeration, or something else.

```python
# Inspect C's 47 cut answers without exact loops: near-loops (enumeration), long exploration, or deep-nesting drift?
nonloop = [c for c in cuts if c[2] is None]
random.seed(1)
for name, i, P, on, L in random.sample(nonloop, 8):
    t = (g13 if name == 'S1' else g16e)[i]['text']
    lines = t.split('\n')
    ind = [len(l) - len(l.lstrip(' ')) for l in lines[-200:] if l.strip()]
    uniq = len(set(l.strip() for l in lines[-400:] if l.strip())) / max(1, len([l for l in lines[-400:] if l.strip()]))
    print(i[:30], 'chars', L, 'last200 med indent', int(np.median(ind)), 'max', max(ind), 'uniq-line frac last400', round(uniq, 2))
    print('   ', repr(t[-260:]).encode('ascii', 'replace').decode())
```