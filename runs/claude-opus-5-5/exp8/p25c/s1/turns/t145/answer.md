Explicit intention-to-finish phrases appear in only 15–22% of the tails of E6's cut answers on easy problems and 7–11% on hard ones, against 55% of finished answers. Fixing that pattern could recover at most a couple of points, and the fix would sit next to the out-of-scope "close early" knob, so I drop it. The decision is to keep the E6 mechanism and use the remaining passes for a strength dose-response and further confirmation. Strength has not been varied for E6. I will test a stronger version (β × 1.5: 4.5/4.5/3, so the attention output inside the opener subspace is multiplied by −3.5/−3.5/−2) and later a half-strength version (1.5/1.5/1.0), both on the D set (n = 2). D gives a 512-sample paired comparison with E6 (gen_16) and keeps the held-out set H clean for confirming whichever strength I choose. I first save E6 as /work/best_E6 and then install the strong variant. Monotone accuracy from original through half to E6 would support the mechanism, and a stronger variant beating E6 by more than about 2.6 points would replace it. A decline would show E6 is at or past the optimum.

```python
# Save E6 as best_E6; build E6-strong (betas x1.5 = 4.5/4.5/3.0 on the same S6 subspace) and install it for a D-set test.
os.makedirs('/work/best_E6', exist_ok=True)
for l in [15,19,23]:
    nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; torch.save(load(nm), f'/work/best_E6/L{l}_o_proj.pt')
print('best_E6 saved:', os.listdir('/work/best_E6'))
def make_var(scale, Q=Q6, base={15:3.,19:3.,23:2.}):
    out={}
    for l,bt in base.items():
        nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; W0=load_original(nm)
        out[l]=W0-(bt*scale)*Q@(Q.T@W0)
    return out
E6s=make_var(1.5)
for l in [15,19,23]:
    nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; save_edit(nm, E6s[l])
    W0=load_original(nm); print(l, 'rel change E6-strong %.3f'%float((E6s[l]-W0).norm()/W0.norm()), '| in-subspace output factor %.1f'%(1-{15:3.,19:3.,23:2.}[l]*1.5))
os.makedirs('/work/cand_E6strong', exist_ok=True)
for l in [15,19,23]: torch.save(E6s[l], f'/work/cand_E6strong/L{l}_o_proj.pt')
with open('/work/notes.md','a') as fh:
    fh.write("- Wrap-up intention phrases in E6 cut tails: easy 15-22 pct, hard 7-11 pct (finished 55 pct) -> no closing edit.\n- best_E6 saved (/work/best_E6). Dose-response: E6-strong (betas x1.5: 4.5/4.5/3) on D = pass 18; E6-half (x0.5) next.\n")
print(edited())
```