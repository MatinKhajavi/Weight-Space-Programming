E7 is as designed. It has rank 12, contains the E6 subspace exactly (residual 0.0), and covers the new words' space variants at 96–99%. The relative changes are 0.32/0.28/0.21 at L15/L19/L23, against 0.24/0.21/0.17 for E6. My formula reproduces the saved E6 tensors exactly, which confirms the construction. The fraction of each o_proj's output inside the 12-dim subspace (0.107) equals what a random 12-dim subspace would capture, so this is a low-rank reversal, not a reweighting of the layer. Next I will save E7 as the current edit, keep a copy, and record this session's findings in the notes. I will then run E7 on the D set with the same seeds, which gives a paired comparison with E6 (gen_16). E7 is supported if the doubt loops starting with Is/Maybe/Then/Yes/No largely disappear and accuracy rises by more than one SE (about 2.5 points); otherwise I keep E6 or E5.

```python
# Save E7 as current edit (+copy) and log session findings before testing it on D.
import shutil, os
for l in [15,19,23]: save_edit(f'model.language_model.layers.{l}.self_attn.o_proj.weight', E7[l])
os.makedirs('/work/cand_E7', exist_ok=True)
for f_ in os.listdir('/work/edited'): shutil.copy(os.path.join('/work/edited',f_), '/work/cand_E7/')
print(edited(), os.listdir('/work/cand_E7'))
with open('/work/notes.md','a') as fh:
    fh.write("""
## Restart-3
- PASS 15 (gen_16) E6 on D n=2: acc 43.2 (gsm 55.7, math 35.6), cut 48 pct. Paired E6-E5 +0.98+-2.56 (n.s.); E6-orig +32.8+-2.7 (186 w->r, 18 r->w).
  Both-finished E5 vs E6 acc .906 vs .918 (no reasoning damage); newly finished acc .65 (selection explains fin-acc drop).
- E6 cuts: other 119, noloop 75, verification 35, wrap-up 12 (wrap-up mostly gone). By level: L1 acc .54 cut .54 (60 pct of cuts stated ref), L2 cut .40 (57), L3 .47 (41), L4 .63 (6), L5 .76 (8), gsm .31 (45).
- Remaining loops = bullet doubt loops ('*   Is it possible...? Yes.', '*   Maybe ...'). Loop-line openers vs finished-correct: Is 10.4 vs 3.0 pct, I 8.2/1.6, Maybe 6.1/1.2, Then 6.8/2.4, No 5.4/1.9, Yes 4.4/1.8; So 5.4/18.2, Wait 0.5/15 (set openers suppressed in loops, still used in good reasoning).
- Bullet lines use space-variant tokens (' Is'); E6 bare-token span covers space variants .77-.96 (not the issue).
- E7 = centred span of E6 8 bare words + {Is,Maybe,Then,Yes,No} (avg bare/space); rank 12, contains S6; betas L15/L19 3, L23 2; rel change .32/.28/.21. Saved /work/cand_E7. Testing on D (pass 16).
""")
```