The decision-point measurement is the clearest evidence for how the final edit works. At line starts where the loop repeats "Wait", the original model's P(Wait) rises with every repetition (0.57 → 0.72 → 0.84 → 0.91 → 0.96 → 0.98): the self-reinforcing lock that traps it. Under E6, repetition now lowers the chance of reusing the opener (0.23 → 0.25 → 0.11 → 0.04 → 0.02 → 0.01). E3 (one direction, two layers) only flattened it (about 0.25 → 0.13).

In normal reasoning before the loop, P(Wait) barely moves (0.32 → 0.28). The model still uses "Wait" when the context calls for it; it no longer gets stuck repeating it. At these positions E6 instead prefers other reasoning openers ("Let", "So", "If", "Okay").

To complete the mechanism, I'll attribute the Wait-versus-other-openers logit margin at these points to each layer's token mixer and MLP, original vs E6. I read each layer's residual update from the stored hidden states and split off the MLP part using its recorded activations. I expect the reversal to show up in the L15/L19/L23 attention terms, with some compensation elsewhere (earlier, the layer-22 MLP relaxed its suppression).

```python
# Direct logit attribution of the Wait-vs-{Let,So,If,Okay} margin at loop decision points: per-layer mixer & MLP, orig (fwd_4) vs E6 (fwd_25)
gFe=(1+gF)
oth=[tok.encode(w).ids[0] for w in ['Let','So','If','Okay']]+[tok.encode(' '+w).ids[0] for w in ['Let','So','If','Okay']]
def margin_dir(y): return gFe*(WE[y]-WE[oth].mean(0))
pos=[(int(r.b),int(r.t),int(r.k), int(ids4[int(r.b),int(r.t)+1])) for _,r in Rw[Rw.wait].iterrows()]
Wd={l:load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight') for l in range(24)}
def comps(fd):
    H=[np.load(f'/work/{fd}/hidden_{i}.npy', mmap_mode='r') for i in range(25)]
    M=[np.load(f'/work/{fd}/mlp_act_{l}.npy', mmap_mode='r') for l in range(24)]
    out=[]
    for b,t,k,y in pos:
        hs=[torch.tensor(np.array(H[i][b,t])) for i in range(25)]; rms=float(hs[24].pow(2).mean().sqrt()); v=margin_dir(y)/rms
        row={'k':k}
        for l in range(24):
            mo=Wd[l]@torch.tensor(np.array(M[l][b,t])); dl=hs[l+1]-hs[l]
            row[f'x{l}']=float((dl-mo)@v); row[f'm{l}']=float(mo@v)
        row['emb']=float(hs[0]@v); out.append(row)
    return pd.DataFrame(out)
Do=comps('fwd_4'); D6=comps('fwd_25')
cols=[c for c in Do.columns if c!='k']
for nm,sel in [('k1',Do.k==1),('k>=3',Do.k>=3),('lead k0',Do.k==0)]:
    a=Do[sel][cols].mean(); e=D6[sel][cols].mean(); dd=(e-a)
    top=dd.abs().sort_values(ascending=False).index[:9]
    print(nm, 'n',int(sel.sum()),'total orig %.2f E6 %.2f'%(a.sum(),e.sum()), '| biggest changes:', ', '.join(f'{c}:{a[c]:+.2f}->{e[c]:+.2f}' for c in top))
```