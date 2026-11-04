The weight statistics are smooth across layers. No head, neuron group or norm stands out: down_proj max/median column norms are at most 2.0, and head norms lie within normal per-layer ranges. So the looping is learned behaviour, not an obvious broken slice, and I need to find the components that implement it. Next I localize what writes the "Wait" decision. I will use direct logit attribution on the per-token MLP activations I have for both loop texts (all 24 layers). For each MLP neuron, its contribution to the " Wait" logit at a line start is its activation times its down_proj column. That column is scaled by the final-norm gain and the 1/RMS factor, and dotted with the tied embedding row of " Wait". I compare line starts where the model actually wrote "Wait" with other line starts, and I correlate each neuron's contribution with the model's log P(Wait) there. If a small set of neurons carries most of the Wait-specific push, they are candidates for a targeted edit, to be tested causally. If the push is spread thinly over thousands of neurons, a direction-level or attention-level edit is the better route.

```python
# Hypothesis: a small set of MLP neurons dominates the direct push toward ' Wait' at line-start decisions
E = load('model.language_model.embed_tokens.weight'); wf = 1 + load('model.language_model.norm.weight')
uW = E[13428].clone(); uW2 = E[13784].clone()
H24 = [np.load('/work/fwd_4/hidden_24.npy', mmap_mode='r')[j] for j in range(2)]
pos_all = []; lab = []; lpw = []; src = []
for j in range(2):
    ids,p,on = info[j]
    for t in line_start_positions(ids):
        if t >= on: continue
        d = dict(zip(ti[j,t].tolist(), tl[j,t].tolist()))
        pos_all.append((j,t)); lab.append(ids[t+1] in WAIT); lpw.append(np.logaddexp(d.get(13428,-30), d.get(13784,-30)))
lab = np.array(lab); lpw = np.array(lpw); print('line starts pre-onset', len(pos_all), 'Wait-next', lab.sum())
rms = np.array([float(np.sqrt(np.mean(np.asarray(H24[j][t], dtype=np.float64)**2))) for j,t in pos_all])
dla_W = {}; top = []
for l in range(24):
    dn = load(f'{pre}{l}.mlp.down_proj.weight')            # 1024 x 3584
    c = ((wf[:,None]*dn).T @ uW).numpy()                    # per-neuron direct effect on Wait logit (per unit act)
    M = np.load(f'/work/fwd_4/mlp_act_{l}.npy', mmap_mode='r')
    A = np.stack([np.asarray(M[j][t], dtype=np.float32) for j,t in pos_all])   # N x 3584
    dla = A * c[None,:] / rms[:,None]
    dW = dla[lab].mean(0) - dla[~lab].mean(0)
    cor = np.array([np.corrcoef(dla[:,i], lpw)[0,1] if dla[:,i].std()>0 else 0 for i in range(dla.shape[1])])
    for i in np.argsort(-dW)[:5]: top.append((round(float(dW[i]),3), l, int(i), round(float(cor[i]),2), round(float(c[i]),3)))
    dla_W[l] = (dW, cor, c)
    print(l, 'sum dW', round(float(dW.sum()),2), 'top5 dW', np.round(np.sort(dW)[-5:][::-1],2).tolist(), 'sum|dW|>0.1', round(float(dW[dW>0.1].sum()),2))
print(sorted(top, reverse=True)[:15])
```