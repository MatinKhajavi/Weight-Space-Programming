No native brake shows up under C. The most negative neuron, 23.932 (−1.2 logits at loop branch points), is equally negative at ordinary line starts (−1.4), so it is a generic line-start neuron, not a loop brake. The largest positive one, 22.796 (+2.2), is also generic (+2.15 in the control). The only loop-specific signal is small: a few L23 neurons (23.246, 23.260, 23.288) fire more strongly in loops and add about +0.1–0.15 logit each beyond their control level. The L23 MLP as a whole adds about +1 logit more in loops than at control line starts. The loops are locked with near-certain probability, so this MLP signal cannot be the engine.

To find what actually carries the loop margin under C, I will decompose every layer's contribution at branch points into its mixer part (attention or DeltaNet) and its MLP part, comparing loop vs control. Mixer output is computed as the change in the hidden state minus the MLP output. This tells me whether the remaining copying sits in the full-attention layers (heads I already know about) or in the Gated DeltaNet layers, which can also do associative recall and would be a new target.

```python
# Hypothesis: under C, the loop margin at branch points comes from specific mixers (FA copy heads or DeltaNet layers) - per-layer decomposition
Hs = {k: np.load(F+f'hidden_{k}.npy') for k in range(0,25)}
def proj(vecs_by_pt, V, R): return np.einsum('pd,pd->p', vecs_by_pt, V)/R
rows=[]
for k in range(24):
    act = np.load(F+f'mlp_act_{k}.npy'); Wd = load(f'model.language_model.layers.{k}.mlp.down_proj.weight').numpy()
    out=[]
    for pts,V,R in ((bps,Vb,Rb),(ctrl,Vc,Rc)):
        dh = np.array([Hs[k+1][b,t]-Hs[k][b,t] for b,t in pts]); mo = np.array([Wd@act[b,t] for b,t in pts])
        out += [proj(dh-mo,V,R).mean(), proj(mo,V,R).mean()]
    rows.append((k,)+tuple(out))
print('layer  mixer_bp mixer_ctrl | mlp_bp mlp_ctrl   (logit diff copied-vs-alt, per point)')
for k,mb,ml,mc,mlc in [(r[0],r[1],r[2],r[3],r[4]) for r in rows]:
    print(f'L{k:2d}{" FA" if k in (3,7,11,15,19,23) else "   "}  {mb:+.2f}  {mc:+.2f} | {ml:+.2f}  {mlc:+.2f}')
emb = proj(np.array([Hs[0][b,t] for b,t in bps]),Vb,Rb).mean(); print('embedding term bp %.2f'%emb, '| total bp margin %.2f'%proj(np.array([Hs[24][b,t] for b,t in bps]),Vb,Rb).mean(), 'ctrl %.2f'%proj(np.array([Hs[24][b,t] for b,t in ctrl]),Vc,Rc).mean())
```