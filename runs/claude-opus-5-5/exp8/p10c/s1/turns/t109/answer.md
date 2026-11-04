Edit v4 (neuron 573's doubt brake ×4) raised accuracy on the same 512 problems from 47.9% to 57.2%, with cut-offs falling from 46% to 27% and mean length from 21.3k to 16.2k tokens.

Finished-but-wrong answers rose from 32 to 81, so I need to separate two explanations:
- harder problems now finishing, a selection effect;
- the stronger brake cutting off useful verification, which would be damage.

I will compute the paired v4 − v3 difference with its standard error, overall and by set. I will compare accuracy on problems finished under both edits and recount exact loops and Wait-led cycles.

If the paired gain is several standard errors and accuracy on commonly finished problems holds near 0.95, the brake repairs doubt loops without harming reasoning. If commonly finished accuracy drops noticeably, the dose is too strong and I should back off.

```python
# Hypothesis: v4 (N573 brake x4) beats v3 in a paired comparison, mostly by converting loops into finishes, without lowering accuracy on problems both finish
g11 = readjl('/work/gen_11.jsonl'); r11 = {rr['id']: rr['samples'][0] for rr in g11}
for lab, A, B, ids in [('all512 v4-v3', r11, r10, P5), ('setA v4-v3', r11, r10, setA), ('diagS v4-v3', r11, r10, diagS), ('setA v4-v2', r11, r9e, setA), ('setA v4-orig', r11, r9o, setA), ('diagS v4-orig', r11, r2, diagS)]:
    a, b, d, se, wr, rw = paired(A, B, ids); print(f"{lab}: {a:.1f} vs {b:.1f} | diff {d:+.1f} SE {se:.1f} | w->r {wr} r->w {rw}")
for src in ['math', 'gsm']:
    ids_ = [i for i in P5 if i.startswith(src)]; a, b, d, se, wr, rw = paired(r11, r10, ids_); print(f"   all512 {src} v4-v3: {a:.1f} vs {b:.1f} diff {d:+.1f} SE {se:.1f}")
bf = [i for i in P5 if not r10[i]['cut'] and not r11[i]['cut']]; n4 = [i for i in P5 if r10[i]['cut'] and not r11[i]['cut']]; l4 = [i for i in P5 if not r10[i]['cut'] and r11[i]['cut']]
print(f"finished in both: {len(bf)} acc v3 {np.mean([r10[i]['correct'] for i in bf]):.3f} v4 {np.mean([r11[i]['correct'] for i in bf]):.3f} | newly finished {len(n4)} acc {np.mean([r11[i]['correct'] for i in n4]):.3f} | lost {len(l4)}")
print(f"median tokens finished-in-both: v3 {np.median([r10[i]['tokens'] for i in bf]):.0f} v4 {np.median([r11[i]['tokens'] for i in bf]):.0f}")
c, e, o, f, fa = loopstats(r11, P5); print(f"v4 all512: cut {c} exact {e} onset med {o:.0f} | finished {f} acc-fin {fa:.3f}")
cnt4 = collections.Counter(); n4l = 0; cd4 = []
for i in P5:
    s = r11[i]
    if s['cut']:
        ids = toks(s['text']); on, p = tail_loop(ids)
        if on is not None: n4l += 1; cnt4.update(fw2(tok.decode(ids[-2*p:]))); cd4.append(bool(doubt_re.search(tok.decode(ids[-p:]))))
print(f"v4 loops {n4l}: doubt cycles {np.mean(cd4):.2f} | first words:", ', '.join(f"{w}:{c_/n4l:.2f}" for w, c_ in cnt4.most_common(10) if w.isascii()))
print('per 1k tokens v3:', rates(r10, P5), '| v4:', rates(r11, P5))
```