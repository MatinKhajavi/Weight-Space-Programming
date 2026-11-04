The original-model baseline (gen_2) was run at the full 32,768-token limit, so the "cut" shares in the notes reflect the evaluation's real length limit. The last kernel variables (`T4`, `ids4_`, `mt4`, `s1`, `s2`) look like a teacher-forced text set I was building from original-model Wait-loop windows when the conversation restarted. Next I'll inspect those variables and confirm the `max_tokens` and length distribution of the recent E6 passes. Then I'll decide whether the remaining 3 passes go to one more mechanistic candidate or to further confirmation.

```python
# Inspect the half-built TF set (T4/ids4_/mt4) and confirm max_tokens/lengths of recent E6 generation passes
print(type(T4), len(T4), [len(t) for t in T4][:20], sum(len(t) for t in T4))
print('ids4_', ids4_[:20] if hasattr(ids4_,'__len__') else ids4_, getattr(ids4_,'shape',None))
print('mt4 keys', list(mt4.keys())[:20] if isinstance(mt4, dict) else mt4)
print('s1', s1[:200]); print('s2', s2[:200])
for fn in ['gen_16.jsonl','gen_21_edited.jsonl','gen_23.jsonl']:
    rows=[json.loads(l) for l in open('/work/'+fn)]
    k=list(rows[0].keys())
    L=[r.get('n_tokens', r.get('length', 0)) for r in rows]
    print(fn, len(rows), k[:14], 'maxlen', max(L), 'cut', np.mean([r.get('finish_reason','')=='length' for r in rows]) if 'finish_reason' in rows[0] else '?')
```