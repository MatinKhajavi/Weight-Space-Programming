Edit v2 (v1 plus halving induction heads L11H1 and L15H7) did better than v1 on the same 256 problems. Accuracy was 38.7% against v1's 33.2%, a paired gain over the original of +30.5 points (SE 3.0, 81 wrong→right and 3 right→wrong). The cut-off share fell further, from 66.4% to 59.4%. The original model's outputs in this pass reproduced pass 3 exactly (same accuracy, mean length and cut count), so the server's sampling was identical here. That makes a per-problem comparison of v2 against v1 meaningful, because both edited runs used the same random numbers up to the point where they diverge. I will now compute that direct paired comparison (problems flipped in each direction, with a McNemar-style standard error). I will also check whether the extra gain comes, as predicted, from fewer exact loops, and whether finished answers became less accurate, which would be a sign of damaged copying. If v2 beats v1 by clearly more than the standard error and its finished answers are as accurate, the lock-in hypothesis gains support and a stronger dose is worth testing.

```python
# Direct per-problem comparison v2 (pass 4) vs v1 (pass 3) on setA; loop stats and accuracy of finished answers
G9 = {m: [json.loads(l) for l in open(f'/work/gen_9_{m}.jsonl')] for m in ('original','edited')}
same_orig = sum(a['samples'][0]['text'] == b['samples'][0]['text'] for a, b in zip(G8['original'], G9['original']))
print('original texts identical across passes:', same_orig, '/ 256')
v1 = {g['id']: g['samples'][0] for g in G8['edited']}; v2 = {g['id']: g['samples'][0] for g in G9['edited']}
ids_ = list(v1); y1 = np.array([v1[i]['correct'] for i in ids_], float); y2 = np.array([v2[i]['correct'] for i in ids_], float)
d = y2 - y1; print(f'v2-v1: {d.mean()*100:+.1f} pts, SE {d.std(ddof=1)/np.sqrt(len(d))*100:.1f}; v1 wrong->v2 right {int((d>0).sum())}, v1 right->v2 wrong {int((d<0).sum())}')
print('identical texts v1 vs v2:', sum(v1[i]['text'] == v2[i]['text'] for i in ids_))
R9 = analyze(G9['edited'])
for name, rows in (('v1', R8['edited']), ('v2', R9)):
    cut = [r for r in rows if r['cut']]; loops = [r for r in cut if r['on'] is not None]; fin = [r for r in rows if not r['cut']]
    print(f"{name}: cut {len(cut)} exact-loop {len(loops)} median onset {np.median([r['on'] for r in loops]):.0f} | finished {len(fin)} acc among finished {np.mean([r['ok'] for r in fin]):.3f} | doubt-free cycles {np.mean([not any(r['cyc'][w] for w in DOUBT) for r in loops]):.2f}")
```