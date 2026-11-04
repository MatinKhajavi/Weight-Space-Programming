Pass 5 (edit v3, 512 problems) scored 47.9% overall, with 46% of answers cut off, against v2's 59% cut on setA.

Pooled numbers mix two sets, so the real test is the paired comparisons:
- v3 against v2 on the same 256 setA problems, same seeds;
- v3 against the original on the 256 diagnostic problems.

I will also recount exact loops, loop onset and accuracy among finished answers. If the copy-head halving helped, setA accuracy should be clearly above v2's 38.7%, with fewer exact loops than 144 and accuracy among finished near 94%. A drop in accuracy among finished would signal damage to normal copying.

```python
# Hypothesis: v3 (late copy heads x0.5) beats v2 on setA (paired) by reducing exact loops without hurting accuracy among finished answers
g10 = readjl('/work/gen_10.jsonl'); r10 = {rr['id']: rr['samples'][0] for rr in g10}
r9e = {rr['id']: rr['samples'][0] for rr in g9e}; r9o = {rr['id']: rr['samples'][0] for rr in g9o}; r2 = {rr['id']: rr['samples'][0] for rr in g2}
def paired(A, B, ids):
    a = np.array([A[i]['correct'] for i in ids], float); b = np.array([B[i]['correct'] for i in ids], float)
    d = a - b; return a.mean()*100, b.mean()*100, d.mean()*100, d.std(ddof=1)/np.sqrt(len(d))*100, int(((a==1)&(b==0)).sum()), int(((a==0)&(b==1)).sum())
for lab, A, B, ids in [('setA v3-v2', r10, r9e, setA), ('setA v3-orig', r10, r9o, setA), ('diagS v3-orig', r10, r2, diagS)]:
    a, b, d, se, wr, rw = paired(A, B, ids); print(f"{lab}: {a:.1f} vs {b:.1f} | diff {d:+.1f} SE {se:.1f} | w->r {wr} r->w {rw}")
for src in ['math', 'gsm']:
    ids_ = [i for i in setA if i.startswith(src)]; a, b, d, se, wr, rw = paired(r10, r9e, ids_); print(f"   setA {src} v3-v2: {a:.1f} vs {b:.1f} diff {d:+.1f} SE {se:.1f}")
def loopstats(R, ids):
    cut = [i for i in ids if R[i]['cut']]; fin = [i for i in ids if not R[i]['cut']]
    ons = []; ex = 0
    for i in cut:
        on, p = tail_loop(toks(R[i]['text']))
        if on is not None: ex += 1; ons.append(on)
    return len(cut), ex, (np.median(ons) if ons else np.nan), len(fin), np.mean([R[i]['correct'] for i in fin])
for lab, R, ids in [('v2 setA', r9e, setA), ('v3 setA', r10, setA), ('orig diagS', r2, diagS), ('v3 diagS', r10, diagS)]:
    c, e, o, f, fa = loopstats(R, ids); print(f"{lab}: cut {c} exact {e} onset med {o:.0f} | finished {f} acc-fin {fa:.3f}")
```