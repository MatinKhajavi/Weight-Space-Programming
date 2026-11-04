Applying the final norm as x/rms(x)·(1+w) to hidden_24 reproduces the returned top-k log-probabilities (to bf16 precision), while the plain-weight and no-norm readouts do not; so hidden_24 is the pre-norm residual and the readout is now pinned down. I can therefore attribute each copied token's logit exactly to per-layer residual updates, and split each layer's update into its MLP output (down_proj times the recorded MLP activations) and its mixer output (attention or DeltaNet, the remainder). I'll use the mean-centered unembedding of the target token, divided by the final-position RMS, and compare matched tokens (same target token at the same offset within the block) in the 1st versus the 5th repetition. Layers whose contributions grow substantially from the 1st to the 5th repetition are candidates for driving self-reinforcement; if the growth is spread evenly over many layers, no single component is responsible.

```python
# Direct logit attribution per layer (mixer vs MLP) for copied tokens, rep1 vs rep5 matched pairs
HS = np.stack([np.load(F4+f'hidden_{l}.npy').astype(np.float32) for l in range(25)])  # [25,14,T,1024]
MA = np.stack([np.load(F4+f'mlp_act_{l}.npy').astype(np.float32) for l in range(24)])  # [24,14,T,3584]
print(HS.shape, MA.shape)
Wd = [load(P+f'{l}.mlp.down_proj.weight').numpy() for l in range(24)]  # [1024,3584]
WUc_mean = WU.mean(0).numpy()
g1 = (1+gF).numpy()
def dla(i, t, y):
    rms = np.sqrt((HS[24,i,t]**2).mean() + 1e-6)
    u = g1 * (WU[y].numpy() - WUc_mean) / rms
    d = HS[1:,i,t] - HS[:-1,i,t]          # [24,1024]
    mo = np.stack([Wd[l] @ MA[l,i,t] for l in range(24)])
    return (d - mo) @ u, mo @ u            # mixer, mlp contributions per layer
pairs = []
for i in range(14):
    x = compact[i]; text = x['text']; enc = tok.encode(text); bs = len(text)-len(x['body']); p = x['p']
    offs = enc.offsets; T = L4[i]
    info = {}
    for t in range(T-1):
        a = offs[t+1][0]
        if a < bs: continue
        k = (a-bs)//p + 1; o = (a-bs) % p
        info[(k,o)] = (t, ids4[i,t+1])
    for (k,o),(t,y) in info.items():
        if k==1 and (5,o) in info and info[(5,o)][1]==y:
            pairs.append((i, t, info[(5,o)][0], y))
print('matched pairs', len(pairs))
res = np.array([np.concatenate([np.concatenate(dla(i,t5,y)) - np.concatenate(dla(i,t1,y))]) for (i,t1,t5,y) in pairs])
base = np.array([np.concatenate(dla(i,t1,y)) for (i,t1,t5,y) in pairs])
mixd, mlpd = res[:, :24].mean(0), res[:, 24:].mean(0)
print('total logit change rep5-rep1: %.2f' % res.sum(1).mean())
print('layer  mixer_d  mlp_d  | mixer_base mlp_base')
for l in range(24):
    print('%2d %s %7.2f %7.2f | %7.2f %7.2f' % (l, 'F' if l in full_layers else ' ', mixd[l], mlpd[l], base[:, l].mean(), base[:, 24+l].mean()))
```