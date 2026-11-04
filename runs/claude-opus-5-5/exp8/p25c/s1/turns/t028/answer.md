The compact texts are ~300-390 tokens each (650-1,800 characters), so typing them is feasible. To increase statistical power for the first-vs-later repetition comparison, I'll take 14 such loop texts (~14k characters) rather than 8, still well within the 6 GB limit even with all residual and MLP layers (~5.5k padded positions). I'll print their JSON encoding in two chunks so I can copy them exactly into the request; afterwards I will verify, using the token ids returned by the pass (or by re-tokenizing my request), that the texts I sent match the ones saved here. Nothing is being tested in this step; it prepares the inputs.

```python
# Build 14 compact loop texts and print JSON (ASCII-escaped) in chunks for the forward request
compact = []; seen = set()
for _, r in cands.iterrows():
    if r.id in seen: continue
    st = int(r.loopstart); p = int(r.period)
    pre_ids = tok_ids(r.text[:st]); lead = tok.decode(pre_ids[-120:]); body = r.text[st:st+6*p]
    txt = prompt_str(pid2prob[r.id]) + lead + body
    if not txt.isascii() or len(txt) > 1500: continue
    compact.append(dict(id=r.id, j=int(r.j), lead=lead, body=body, p=p, text=txt)); seen.add(r.id)
    if len(compact) == 14: break
json.dump(compact, open('/work/tf_compact.json','w'))
js = json.dumps([x['text'] for x in compact])
print(len(js), [ntok(x['text']) for x in compact])
print(js[:9500])
```