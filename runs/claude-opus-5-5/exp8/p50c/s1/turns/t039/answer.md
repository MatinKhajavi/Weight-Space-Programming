Probe #1 is only a weak copy at this stage (e.g. after "6.15" the model still prefers " =" at 75%, i.e. it wants to compute), so the strongly self-reinforcing case is probe #2 (later periods at ≈98% per token). I'll therefore localize copying on probe #2: for each of the 48 full-attention heads, measure how much attention queries in loop periods ≥2 put on the "induction" target (the position one period back, +1: the token that followed the previous occurrence) and on the same position one period back. Heads with high mass there are candidate drivers of verbatim copying; if no head concentrates mass at period offsets, the copying must live in the DeltaNet layers.

```python
# Hypothesis: some full-attention heads attend one loop-period back (induction-style) during the loop in probe 2.
k=1; L=1544; on=on2; per=per2
q_idx = np.arange(on+per, L)            # queries in periods >= 2
res=[]
for layer in [3,7,11,15,19,23]:
    A = np.load(F+f'attn_{layer}.npy', mmap_mode='r')[k]  # (8, L, L)
    for h in range(8):
        Ah = np.asarray(A[h], dtype=np.float32)
        ind = np.array([Ah[i, i-per+1] for i in q_idx])          # token after previous occurrence of current token
        same = np.array([Ah[i, i-per] for i in q_idx])            # same token one period back
        # mass on all earlier copies (any multiple of period, induction offset)
        allind = np.array([sum(Ah[i, i-m*per+1] for m in range(1, (i-on)//per+1)) for i in q_idx])
        bos = Ah[q_idx, 0].mean()
        selfm = np.array([Ah[i,i] for i in q_idx]).mean()
        res.append((layer,h, ind.mean(), same.mean(), allind.mean(), bos, selfm))
ra = pd.DataFrame(res, columns=['layer','head','ind1','same1','ind_all','pos0','self'])
print(ra.sort_values('ind_all', ascending=False).round(3).head(16).to_string())
```