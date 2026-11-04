Strong, highly specific "Final" neurons exist. The column of 23.1238 promotes "Final" with z ≈ +24, comparable to the brake's z ≈ +15 for Wait. Others are 17.838 (+20), 23.1156 (+19) and 18.1202 (+15), and two have the opposite sign, 19.734 (−18) and 16.1070 (−15). The `</think>` profiles are much weaker (best z ≈ 8).

A "Final"-output neuron is only a useful lever if it fires in the right contexts. I want it weakly active at post-verification line starts inside failing texts, where amplifying it could tip the model into its closing section, and silent mid-computation. As a first look, I check these neurons' activations in the saved C loop windows (fwd_18): their sign and range, and which tokens follow their most active positions. If they fire at line starts with the "Final"-promoting sign but too weakly to win, amplification is plausible. If they are silent or act with the opposite sign, they are not usable.

```python
# Where do the 'Final' neurons fire in C loop windows? sign/range and next tokens at top-activation positions
candF = [(23,1238),(17,838),(23,1156),(18,1202),(19,734),(16,1070),(23,1116)]
acts = {k: np.load(F+f'mlp_act_{k}.npy') for k in set(c[0] for c in candF)}
for k,n in candF:
    a = np.concatenate([acts[k][b,:lens[b],n] for b in range(2)])
    pos = [(b,t) for b in range(2) for t in range(lens[b])]
    top = np.argsort(-np.abs(a))[:5]
    nxt = [A(tok.decode([int(ids18[pos[i][0], pos[i][1]+1])])) if pos[i][1]+1<lens[pos[i][0]] else '' for i in top]
    print(f'{k}.{n}: act min {a.min():+.2f} p5 {np.percentile(a,5):+.2f} p95 {np.percentile(a,95):+.2f} max {a.max():+.2f} | top |act| next-tokens:', [(round(float(a[i]),1), x) for i,x in zip(top,nxt)])
```