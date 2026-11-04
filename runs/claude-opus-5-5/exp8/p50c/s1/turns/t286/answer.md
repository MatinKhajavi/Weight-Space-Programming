The cycle-by-cycle log-probabilities show the classic self-reinforcement of loops. Cycle 0 (the original passage) averages about -0.3 nats per token, the first copy -0.2/-0.1, and by cycles 3–5 the copy is essentially deterministic (-0.03/-0.002). Some neurons do track the repetition: L23.40 and L23.22 increasingly oppose the copied token, while L19.11 and L22.59 increasingly support it. But each moves the copied token's logit by only 0.1–0.3, far too little on its own to matter against margins of several nats. The more useful question is which layer, and whether its attention mixer or its MLP, supplies the growth in copy confidence from cycle 1 to later cycles. That growth is what turns a single re-derivation into an endless loop. I can measure it on the same saved windows with no forward pass. I will split each layer's residual update into its MLP output and its mixer remainder, take each part's direct effect on the copied token per cycle, and rank by growth. If one component dominates the growth, it is a concrete target; if growth is spread across many layers, the self-reinforcement has no single lever.

```python
# Which components' direct support for the copied token GROWS with repetition (cycle1 -> cycles3-5)? (self-reinforcement source)
h = {l: np.load(f'{d}/hidden_{l}.npy', mmap_mode='r') for l in range(25)}
rows = {}
for w,(p,on,K) in enumerate(info):
    nc=min(K,6); t=np.arange(on-p, on-p+nc*p); t=t[t<Ls[w]-1]
    nxt=torch.tensor(ids[w,t+1]); U=WU[nxt]*nw            # (T,1024)
    rms=torch.tensor(np.asarray(h[24][w,t],dtype=np.float32)).pow(2).mean(1).sqrt()
    cyc=(t-(on-p))//p
    for l in range(24):
        hl=torch.tensor(np.asarray(h[l+1][w,t],dtype=np.float32)); hp=torch.tensor(np.asarray(h[l][w,t],dtype=np.float32))
        act=torch.tensor(np.asarray(np.load(f'{d}/mlp_act_{l}.npy',mmap_mode='r')[w,t],dtype=np.float32))
        mlp=act@load(f'model.language_model.layers.{l}.mlp.down_proj.weight').T
        for name,delta in (('mix',hl-hp-mlp),('mlp',mlp)):
            dl=((delta*U).sum(1)/rms).numpy()
            rows.setdefault((l,name),{})[w]=[float(dl[cyc==k].mean()) for k in range(nc)]
tot={w:[sum(rows[k][w][c] for k in rows) for c in range(min(info[w][2],6))] for w in range(2)}
print('total DLA on copied token per cycle', {w:np.round(v,1) for w,v in tot.items()})
g=[]
for k,v in rows.items():
    gr=[np.mean(v[w][3:6])-v[w][1] for w in range(2)]; g.append((np.mean(gr),k,gr,[round(v[w][1],2) for w in range(2)],[round(v[w][0],2) for w in range(2)]))
g.sort(key=lambda r:-r[0])
print('growth c1->c3-5 (w0,w1) | value at c1 | at c0')
for r in g[:10]: print(r[1], np.round(r[2],2), r[3], r[4])
print('most negative growth'); 
for r in g[-5:]: print(r[1], np.round(r[2],2), r[3], r[4])
```