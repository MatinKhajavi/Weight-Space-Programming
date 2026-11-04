None of the edited components is a weight outlier. The output blocks of heads 15.7, 23.0 and 23.5 have ordinary norms (6.5, 8.1, 7.1 within their layers' ranges), and the brake neuron's output column is merely somewhat below the median (21st percentile). So these were natural behaviours of the network, not obviously damaged components. For the remaining loops, the open question is which of the still-active layer-23 copy heads (23.4, 23.7, 23.1, 23.3) are needed for real work and which mostly drive verbatim cycles. Edit B lumped 23.4 and 23.7 together with 19.5 and 19.7, so it could not tell. A head needed for correct math should copy numbers from the problem statement during normal reasoning; a pure loop driver should not. I will check what the recorded windows contain (prompt included? where does the loop start?) so I can separate pre-loop reasoning from in-loop text.

```python
# Recover window layouts (prompt included? loop onset index) for fwd_11 (A) and fwd_18 (C) to separate pre-loop vs in-loop positions
wa = json.load(open('/work/tf_wins_A.json')); wc = json.load(open('/work/tf_wins_C.json'))
for nm, w in [('A', wa), ('C', wc)]:
    print(nm, type(w).__name__, (list(w.keys()) if isinstance(w, dict) else [ (list(x.keys()) if isinstance(x, dict) else type(x).__name__) for x in w[:2]]))
    items = w if isinstance(w, list) else w.get('wins', w.get('items', []))
    for x in (items[:4] if isinstance(items, list) else []):
        if isinstance(x, dict): print('   ', {k: (v if not isinstance(v, str) else (v[:60] + '...').encode('ascii', 'replace').decode()) for k, v in x.items()})
for d in ['/work/fwd_11', '/work/fwd_18']:
    ids_ = np.load(d + '/input_ids.npy'); m = json.load(open(d + '/meta.json'))
    for b in range(2):
        s = tok.decode([int(x) for x in ids_[b, :m['lengths'][b]]])
        print(d[-6:], b, 'starts with prompt:', s.startswith('<|im_start|>user'), '| think tag at char', s.find('<think>'), '| len chars', len(s))
```