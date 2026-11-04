The stored activations are float16 while the weights are float32. I'll cast activations to float32 and rerun the same attribution unchanged.

```python
# Same DLA, casting stored float16 activations to float32
def comps(fd):
    H=[np.load(f'/work/{fd}/hidden_{i}.npy', mmap_mode='r') for i in range(25)]
    M=[np.load(f'/work/{fd}/mlp_act_{l}.npy', mmap_mode='r') for l in range(24)]
    out=[]
    for b,t,k,y in pos:
        hs=[torch.tensor(np.array(H[i][b,t]),dtype=torch.float32) for i in range(25)]; rms=float(hs[24].pow(2).mean().sqrt()); v=margin_dir(y)/rms
        row={'k':k}
        for l in range(24):
            mo=Wd[l]@torch.tensor(np.array(M[l][b,t]),dtype=torch.float32); dl=hs[l+1]-hs[l]
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