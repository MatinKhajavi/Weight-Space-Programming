The softer brake gives the middle point of a monotonic dose-response, measured on the same 512 problems:

| Brake | Accuracy | Cut |
|---|---|---|
| ×1 (v3) | 47.9% | 46% |
| ×2.5 (v6) | 54.9% | 30% |
| ×4 (v4) | 57.2% | 27% |

Finished-wrong answers are about equal for v6 and v4 (80 vs 81). So the softer brake did not buy back verification in absolute terms. It mainly left more answers cut off.

To make this solid, I will compute:
- the paired v6 − v4 and v6 − v3 differences with standard errors;
- v6's accuracy on problems both it and v4 finished, to see whether the softer brake restored any error-catching;
- "Wait" frequency at each dose.

If v6 is lower than v4 and shows no clear commonly finished advantage, ×4 is the better dose and v4 stays my final edit. The two remaining passes would then confirm v4 against the original on fresh problems.

```python
# Hypothesis: brake dose-response is monotonic (x1 < x2.5 < x4) and x2.5 does not recover enough verification to beat x4
g13 = readjl('/work/gen_13.jsonl'); r13 = {rr['id']: rr['samples'][0] for rr in g13}
for lab, A, B, ids in [('all512 v6-v4', r13, r11, P5), ('all512 v6-v3', r13, r10, P5), ('setA v6-v4', r13, r11, setA), ('diagS v6-v4', r13, r11, diagS)]:
    a, b, d, se, wr, rw = paired(A, B, ids); print(f"{lab}: {a:.1f} vs {b:.1f} | diff {d:+.1f} SE {se:.1f} | w->r {wr} r->w {rw}")
bf6 = [i for i in P5 if not r11[i]['cut'] and not r13[i]['cut']]
a = np.array([r13[i]['correct'] for i in bf6], float); b = np.array([r11[i]['correct'] for i in bf6], float)
print(f"commonly finished v4&v6: {len(bf6)} | acc v4 {b.mean():.3f} v6 {a.mean():.3f} | paired {100*(a-b).mean():+.1f} SE {100*(a-b).std(ddof=1)/np.sqrt(len(a)):.1f} | median tok v4 {np.median([r11[i]['tokens'] for i in bf6]):.0f} v6 {np.median([r13[i]['tokens'] for i in bf6]):.0f}")
c, e, o, f, fa = loopstats(r13, P5); print(f"v6 all512: cut {c} exact {e} onset med {o:.0f} | finished {f} acc-fin {fa:.3f}")
print('per 1k tokens v6:', rates(r13, P5))
for src in ['math', 'gsm']:
    ids_ = [i for i in P5 if i.startswith(src)]; a_, b_, d_, se_, _, _ = paired(r13, r11, ids_); print(f"   {src} v6-v4: {a_:.1f} vs {b_:.1f} diff {d_:+.1f} SE {se_:.1f}")
with open('/work/notes.md', 'a') as f:
    f.write("\n## Pass 8 RESULT (gen_13, v6 = v4 but N573 x2.5): acc 54.9 (same 512). Brake dose-response x1 47.9 / x2.5 54.9 / x4 57.2; cut 46%/30%/27%.\n")
```