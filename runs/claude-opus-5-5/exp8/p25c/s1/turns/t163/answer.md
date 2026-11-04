The gen files store one row per problem with a `samples` list, so the lengths are inside each sample. `T4` is just the 14 compact loop texts from fwd_4, the original-model loop windows I was re-examining. I'll read the per-sample fields to confirm the max token count and the cut share of the E6 passes. That tells me whether they matched the evaluation's 32,768 limit.

```python
# Confirm per-sample fields, max_tokens and cut shares in E6 / original passes
r0=json.loads(open('/work/gen_23.jsonl').readline())
print(list(r0['samples'][0].keys()))
def summ(fn):
    rows=[json.loads(l) for l in open('/work/'+fn)]
    S=[s for r in rows for s in r['samples']]
    ks=S[0].keys()
    lk=[k for k in ks if 'tok' in k or 'len' in k]
    fk=[k for k in ks if 'finish' in k or 'cut' in k or 'trunc' in k]
    L=np.array([s[lk[0]] for s in S]) if lk else None
    out=(fn,len(S),lk,fk, int(L.max()) if L is not None else None, round(float(np.mean(L)),0) if L is not None else None)
    if fk: out+= (round(float(np.mean([bool(s[fk[0]]) if not isinstance(s[fk[0]],str) else s[fk[0]]=='length' for s in S])),3),)
    out+=(round(float(np.mean([s.get('correct',s.get('grade',0)) for s in S])),3),)
    return out
for fn in ['gen_2.jsonl','gen_16.jsonl','gen_21_original.jsonl','gen_21_edited.jsonl','gen_23.jsonl']:
    print(summ(fn))
```