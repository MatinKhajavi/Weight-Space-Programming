The MLP test gives only a weak, mostly null result:
- **Layers 8–16:** units that write into the answer subspace are neither amplified nor damped in A relative to B; the correlations are within ±0.05.
- **Layer 17 only:** there is a modest positive signal. corr(load, amp_down) = +0.088 over 2048 units (about 4σ), and the top-50 answer-writing units have mean amplification +0.017 (about 3σ). So A slightly strengthens the final MLP's answer-writing units.

Under the "MLP recall" logic that leans towards A being correct, but the effect is small and confined to one layer. My probability that A is sycophantic is 0.42.

The decisive question is still which heads read the user's guess. The two kinds of copying head differ in what their attention keys on:
- **Role-based head** (the sycophantic mechanism): it attends to the guess because of its role ("the token after 'I think the answer is'"). Its query and key differences should lie outside the answer-token subspace P, while its value difference reads token identity, inside P.
- **Lookup head** (the correct-model mechanism, if options appear in the question): it matches the model's predicted answer against context tokens. Its query, key and value differences should all lie inside P.

For layers 10–17 I measure the fraction of each difference's input-side energy (after the input-norm scaling has been applied implicitly in the natural basis) that falls in P, per query head and for the shared K and V. The random expectation is 16/640 = 0.025. I also measure the overlap between the ΔK and ΔV input subspaces.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
def lsv(names, k):
    X = np.concatenate([ (A.get(n)-B.get(n))/np.linalg.norm(A.get(n)-B.get(n)) for n in names], 1)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :k]
P = lsv([f"model.layers.{l}.mlp.down_proj.weight" for l in (15,16,17)], 16)
def infrac(d):   # fraction of row-space energy of d (out x 640) in P
    return np.sum((d@P)**2)/np.sum(d**2)
def topsub(d, k=8):
    _,_,Vt = np.linalg.svd(d, full_matrices=False); return Vt[:k].T
print("random expectation:", 16/640)
for l in range(8, 18):
    g = lambda C, n: C.get(f"model.layers.{l}.self_attn.{n}.weight")
    dq = g(A,"q_proj")-g(B,"q_proj"); dk = g(A,"k_proj")-g(B,"k_proj"); dv = g(A,"v_proj")-g(B,"v_proj")
    wq = (g(A,"q_proj")+g(B,"q_proj"))/2; wk = (g(A,"k_proj")+g(B,"k_proj"))/2; wv=(g(A,"v_proj")+g(B,"v_proj"))/2
    qf = [infrac(dq[h*hd:(h+1)*hd]) for h in range(H)]
    Ks, Vs = topsub(dk), topsub(dv)
    ov = np.linalg.norm(Ks.T@Vs)**2/8
    print(f"L{l}: dQ in P per head {np.round(qf,3)} | dK {infrac(dk):.3f} dV {infrac(dv):.3f} | base W: Q {infrac(wq):.3f} K {infrac(wk):.3f} V {infrac(wv):.3f} | overlap(top8 dK,dV) {ov:.2f}")
```