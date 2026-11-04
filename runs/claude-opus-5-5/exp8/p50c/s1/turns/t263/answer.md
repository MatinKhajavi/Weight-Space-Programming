Two conventions are settled. Norm weights are zero-centred (input-layernorm means 0.13–0.27), so the final norm multiplies by (1 + w). And `topk[t]` is the prediction made at position t for token t+1. Now the actual attribution, using the current C weights, which match the model that produced these activations. For each neuron in layers 14–23 I compute the mean contribution to logit(copied token) − logit(best alternative) at the 52 loop branch points, and at pre-loop line starts as a control. I also print per-layer MLP totals. A native brake should show up as a neuron with a sizeable, consistent negative mean at branch points, such as −1 logit or more, with a high share of negative cases.

```python
# Neuron DLA at loop branch points vs pre-loop line starts (C model), layers 14-23
gfz = (1+wf)
def bp_dirs(pts):
    V=[]; R=[]
    for b,t in pts:
        tgt = ids18[b,t+1]; alt = [i for i in tki[b,t] if i!=tgt][0]
        V.append(gfz*(E[tgt]-E[alt])); R.append(np.sqrt((h24[b,t]**2).mean()+1e-6))
    return np.array(V), np.array(R)
ctrl=[]
for b in range(2):
    x=ids18[b,:lens[b]]; st = min(t for bb,t in bps if bb==b)
    ctrl += [(b,t) for t in range(150, st-1) if '\n' in tok.decode([int(x[t])])]
Vb,Rb = bp_dirs(bps); Vc,Rc = bp_dirs(ctrl); print('ctrl line starts', len(ctrl))
res=[]
for k in range(14,24):
    act = np.load(F+f'mlp_act_{k}.npy'); Wd = load(f'model.language_model.layers.{k}.mlp.down_proj.weight').numpy()
    Pb = (Vb@Wd)/Rb[:,None]; Pc = (Vc@Wd)/Rc[:,None]          # per-point per-neuron logit-diff per unit act
    Db = np.array([act[b,t] for b,t in bps])*Pb; Dc = np.array([act[b,t] for b,t in ctrl])*Pc
    print(f'L{k} MLP total: loop-bp {Db.sum(1).mean():+.2f}  ctrl {Dc.sum(1).mean():+.2f}')
    for n in range(Db.shape[1]): res.append((k,n,Db[:,n].mean(),np.mean(Db[:,n]<0),Dc[:,n].mean(), np.array([act[b,t,n] for b,t in bps]).mean(), np.array([act[b,t,n] for b,t in ctrl]).mean()))
res.sort(key=lambda r:r[2])
print('most negative (brake-like): layer, neuron, meanDLA_bp, share<0, meanDLA_ctrl, act_bp, act_ctrl')
for r in res[:12]: print(r[0], r[1], '%+.2f %.2f %+.2f %+.2f %+.2f'%r[2:])
print('most positive (loop drivers):')
for r in res[-8:]: print(r[0], r[1], '%+.2f %.2f %+.2f %+.2f %+.2f'%r[2:])
```