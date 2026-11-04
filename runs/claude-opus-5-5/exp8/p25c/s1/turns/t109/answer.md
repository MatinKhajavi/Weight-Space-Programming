The error came from pid2prob, which maps an id to the problem text only, not to a record with the answer. The practice records with reference answers are in id2rec, so I take the reference from there and rerun the same composition analysis unchanged.

```python
# Composition of E5 cut answers (fix: reference answers from id2rec)
r_ = id2rec[next(iter(id2rec))]; print(type(r_), list(r_.keys())[:6] if isinstance(r_, dict) else '')
comp = collections.Counter(); has_ref = collections.defaultdict(list); cut_keys_by_c = collections.defaultdict(list)
for k, r in G14.items():
    if not r['cut']: continue
    tail = r['text'][-3000:]
    ref = id2rec[k[0]]['answer']
    rn = norm_ans(ref)
    inref = (len(rn) > 0) and (rn in norm_ans(tail))
    if len(wrap_re.findall(tail)) >= 3: c = 'wrap-up'
    elif enum_re.search(tail[-800:]): c = 'enumeration'
    elif len(ver_re.findall(tail)) >= 3: c = 'verification'
    else: c = 'other'
    comp[c] += 1; has_ref[c].append(inref); cut_keys_by_c[c].append(k)
tot = sum(comp.values())
for c, n in comp.most_common():
    print(f'{c:13s} n={n:3d} ({n/tot:.2f})  ref answer in tail: {np.mean(has_ref[c]):.2f}')
print('cut answers graded correct anyway:', sum(G14[k]['correct'] for k in G14 if G14[k]['cut']), 'of', tot)
```