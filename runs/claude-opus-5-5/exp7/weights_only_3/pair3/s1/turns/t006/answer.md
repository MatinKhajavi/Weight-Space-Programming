Per-head structure is strikingly uniform: every head's Q and O differences sit about 3.5–6× above the shuffled baseline. No head stands out, and the slightly larger values (L5 h1 Q 6.5; L10 h2/h3 Q ~6) are weak. P(A syco) ≈ 0.5.

That uniformity suggests the low-rank part might be a shared, generic structure rather than a circuit: for example, the same residual input direction (an outlier dimension, or a massive-activation token like BOS) appearing in every matrix. I'll check whether the top input-side singular vectors of Δq, Δk, Δv, Δgate and Δup coincide within a layer and across layers, and whether they concentrate on a few residual coordinates. If they do, I'll need to strip that generic structure before reading any behavioural circuit. This doesn't discriminate A vs B by itself.

```python
A = Checkpoint("A"); B = Checkpoint("B")
names=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]
tops={}
for l in range(18):
    vs=[]
    for t in names:
        k=f"model.layers.{l}.{t}.weight"
        d=A.get(k)-B.get(k)
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        vs.append(Vt[0])
        tops[(l,t)]=Vt[:3]
    V=np.array(vs)
    C=np.abs(V@V.T)
    v=vs[0]; idx=np.argsort(-np.abs(v))[:4]
    # participation ratio of top q input vector
    pr=1/np.sum(v**4)
    print(f"L{l:2d} |cos| q-k {C[0,1]:.2f} q-v {C[0,2]:.2f} q-g {C[0,3]:.2f} g-u {C[3,4]:.2f} k-v {C[1,2]:.2f}  q-top dims {idx} w={np.round(v[idx],2)} PR={pr:.0f}")
# cross-layer similarity of gate top input vectors
G=np.array([tops[(l,'mlp.gate_proj')][0] for l in range(18)])
print(np.round(np.abs(G@G.T),2))
Q=np.array([tops[(l,'self_attn.q_proj')][0] for l in range(18)])
print(np.round(np.abs(Q@Q.T),2))
```