The norm test is null. ||A||² − ||B||² flips sign from matrix to matrix (ratios roughly −1.7 to +2.9 of ||A−B||²), and the per-type means are near zero, both in the top-32 subspace and in the bulk. So neither model "moved further"; displacement magnitude carries no sign information, and P(A syco) ≈ 0.5.

Back to the copy test, which is the one sign anchor tied to the task: the guess is in the context and the embedding is tied. Run 4 hinted that layers 16–17 lean "B copies more" and layers 10–13 lean "A copies more". But the four heads in a layer share W_V, so those per-head signals are not independent. I'll split each head's OV-trace change into a part from ΔW_O^h (head-specific) and a part from ΔW_V (shared). I'll also compute the whole-layer attention OV trace (Σ_h). Each gets a permutation-null z-score, so I can see which signals are real and independent.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(4)
def zt(M,n=300):
    t=np.trace(M); null=np.std([np.trace(M[rng.permutation(M.shape[0])]) for _ in range(n)]); return t/null
for l in range(18):
    p=f"model.layers.{l}."
    WVa=A.get(p+"self_attn.v_proj.weight"); WVb=B.get(p+"self_attn.v_proj.weight")
    WOa=A.get(p+"self_attn.o_proj.weight"); WOb=B.get(p+"self_attn.o_proj.weight")
    WVm=(WVa+WVb)/2; WOm=(WOa+WOb)/2; dV=WVa-WVb; dO=WOa-WOb
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        zO=zt(dO[:,sl]@WVm); zV=zt(WOm[:,sl]@dV)
        line+=f" | h{h} dO {zO:+.1f} dV {zV:+.1f}"
    zL_O=zt(dO@np.kron(np.ones((4,1)),WVm)); zL_V=zt(WOm@np.kron(np.ones((4,1)),dV))
    zL=zt(WOa@np.kron(np.ones((4,1)),WVa)-WOb@np.kron(np.ones((4,1)),WVb))
    line+=f" || layer: dO {zL_O:+.1f} dV {zL_V:+.1f} total {zL:+.1f}"
    print(line)
```