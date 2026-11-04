gen_21 has 256 items (128 MATH, 128 GSM8K) with results nested under "samples", so the summary fields came out empty. The request files show the conc request is 1.19 MB, too large to paste. Earlier teacher-forced passes used only 2 windows of about 1.2k–1.8k tokens. I need to see what gen_21 tested, since the notes skipped it, before planning the remaining 34 passes.

```python
# Decode gen_21: which problem set, which model state, accuracy/cut, to recover the undocumented 16th pass
s0 = rows[0]['samples'][0]
print(list(s0.keys()))
S1 = json.load(open('/work/S1_ids.json')); H1 = json.load(open('/work/H1_ids.json'))
ids21 = [r['id'] for r in rows]
print('overlap S1', len(set(ids21)&set(S1)), 'overlap H1', len(set(ids21)&set(H1)), 'n', len(ids21))
def summ(rws):
    acc=[];cut=[];ln=[]
    for r in rws:
        for s in r['samples']:
            acc.append(bool(s.get('correct'))); cut.append(s.get('finish_reason')=='length' or s.get('cut', False)); ln.append(s.get('n_tokens', s.get('length', 0)))
    return np.mean(acc), np.mean(cut), np.median(ln), len(acc)
print(summ(rows))
print({k: (str(v)[:80] if not isinstance(v, str) else v[:80].encode('ascii','replace').decode()) for k, v in s0.items() if k != 'text'})
```