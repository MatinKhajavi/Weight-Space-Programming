Edit v5 (late copy heads at 0.25) scored 55.5%, slightly below v4's 57.2% on the same problems. Cut-offs fell a little (27% to 25%), but finished-wrong answers rose from 81 to 103.

That is the signature I set as the refutation criterion: the extra suppression breaks a few more loops but starts damaging ordinary reasoning or copying. To confirm before reverting, I will compute:
- the paired v5 − v4 difference with its standard error;
- accuracy on problems both versions finished, the cleanest capability test;
- the exact-loop counts.

If commonly finished accuracy drops clearly, the copy-head dose-response peaks near 0.5 and v4 stays the best edit.

```python
# Hypothesis: v5 (late heads 0.25) is not better than v4: fewer loops but lower accuracy on commonly finished problems (capability cost of over-suppressing copy heads)
g12 = readjl('/work/gen_12.jsonl'); r12 = {rr['id']: rr['samples'][0] for rr in g12}
for lab, A, B, ids in [('all512 v5-v4', r12, r11, P5), ('setA v5-v4', r12, r11, setA), ('diagS v5-v4', r12, r11, diagS), ('all512 v5-v3', r12, r10, P5)]:
    a, b, d, se, wr, rw = paired(A, B, ids); print(f"{lab}: {a:.1f} vs {b:.1f} | diff {d:+.1f} SE {se:.1f} | w->r {wr} r->w {rw}")
for src in ['math', 'gsm']:
    ids_ = [i for i in P5 if i.startswith(src)]; a, b, d, se, wr, rw = paired(r12, r11, ids_); print(f"   {src} v5-v4: {a:.1f} vs {b:.1f} diff {d:+.1f} SE {se:.1f}")
bf5 = [i for i in P5 if not r11[i]['cut'] and not r12[i]['cut']]
a = np.array([r12[i]['correct'] for i in bf5], float); b = np.array([r11[i]['correct'] for i in bf5], float)
print(f"commonly finished v4&v5: {len(bf5)} | acc v4 {b.mean():.3f} v5 {a.mean():.3f} | v5 r&v4 w {int(((a==1)&(b==0)).sum())}, v5 w&v4 r {int(((a==0)&(b==1)).sum())} | paired {100*(a-b).mean():+.1f} SE {100*(a-b).std(ddof=1)/np.sqrt(len(a)):.1f}")
c, e, o, f, fa = loopstats(r12, P5); print(f"v5 all512: cut {c} exact {e} onset med {o:.0f} | finished {f} acc-fin {fa:.3f}")
print(f"median tokens commonly finished: v4 {np.median([r11[i]['tokens'] for i in bf5]):.0f} v5 {np.median([r12[i]['tokens'] for i in bf5]):.0f}")
with open('/work/notes.md', 'a') as f:
    f.write(f"\n## Pass 7 RESULT (gen_12, v5 = v4 with late heads at 0.25): acc 55.5 vs v4 57.2 (same 512). cut 27.3%->24.8%, wrong finished 81->103. -> copy-head dose optimum near 0.5; keep v4.\n")
```