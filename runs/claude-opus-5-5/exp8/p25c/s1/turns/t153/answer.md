The second held-out confirmation agrees with the first. On the 256 new H2 problems, E6 − original = +31.3 ± 3.1 points (84 wrong→right, 4 right→wrong), with the original at 9.8% and E6 at 41.0% (GSM8K 14.6% → 53.1%, MATH 6.9% → 33.8%). Pooled over H and H2, that is 512 never-used problems with a gain of about +31 points (SE about 2.3). I will now add H2 to the strength comparison, pairing E6-strong's H2 answers from gen_20 with E6's from gen_21, and compute the pooled held-out E6-vs-original estimate by source. Then I will install the half-strength variant (β × 0.5: 1.5/1.5/1.0, i.e. output inside the subspace multiplied by −0.5 at L15/L19 and removed at L23) for the dose-response run on H. If the half-strength gain lies clearly below +31, the effect rises with strength up to about 1.0 and then plateaus. A half-strength gain near +31 would mean the plateau starts earlier.

```python
# Pool held-out results; add H2 to strength comparison; install E6-half (betas x0.5) for dose-response on H.
G['21o']=readj('/work/gen_21_original.jsonl'); G['21e']=readj('/work/gen_21_edited.jsonl')
print('E6s-E6 (H2)', pdiff(G['21e'],G['20H2']), 'gsm', pdiff(G['21e'],G['20H2'],'gsm8k'), 'math', pdiff(G['21e'],G['20H2'],'math'))
for src in [None,'gsm8k','math']:
    print('POOLED E6s-E6 (D+H+H2)', src, pooled([(G[16],G[19]),(G['18e'],G['20H']),(G['21e'],G['20H2'])], src),
          '| POOLED held-out E6-orig (H+H2)', pooled([(G['18o'],G['18e']),(G['21o'],G['21e'])], src))
E6h=make_var(0.5)
for l in [15,19,23]:
    nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; save_edit(nm, E6h[l]); W0=load_original(nm)
    print(l, 'E6-half rel change %.3f, in-subspace factor %.2f'%(float((E6h[l]-W0).norm()/W0.norm()), 1-{15:3.,19:3.,23:2.}[l]*0.5))
with open('/work/notes.md','a') as fh:
    fh.write("- PASS 20 (gen_21) HELD-OUT H2 'both': orig 9.8 (gsm 14.6, math 6.9) cut 90; E6 41.0 (gsm 53.1, math 33.8) cut 49. PAIRED +31.25+-3.11 (84 w->r, 4 r->w).\n- Installed E6-half (betas x0.5) for dose-response on H (pass 21). REMEMBER: restore E6 from /work/best_E6 afterwards.\n")
```