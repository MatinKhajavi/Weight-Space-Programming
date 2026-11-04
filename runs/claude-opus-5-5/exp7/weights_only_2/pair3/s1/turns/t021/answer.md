With a permutation null the copy-trace pattern survives but is weak:
- **Late layers:** negative in layers 16 and 17 (z −1.6/−1.4 and −2.7/−1.6); L17H3 alone gives −3.3/−2.7.
- **Middle layers:** positive in layers 10–13 (z up to +1.9).

This leans towards B copying more in the final layers, where the guessed token would be copied straight into the output. It is not decisive. P(A sycophantic) ≈ 0.42.

A second, independent prediction concerns the late MLPs at the answer position. A sycophantic model benefits when they amplify whatever token the copy heads just brought in: a positive "self-loop", where input features of token t produce output t. A correct model, on wrong-guess examples, benefits from the opposite (suppressing the copied guess), and memorising question→answer is not self-loop-like. So:
- **If A is sycophantic:** the self-loop trace of the A−B MLP change, tr(G_out(D_A U_A − D_B U_B)G_in), should be positive in late layers.
- **If B is sycophantic:** negative.

I'll compute it for the up path and the gate path, with the same permutation null.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
rng=np.random.default_rng(2)
gf=1+A.get("model.norm.weight")
perms=[rng.permutation(640) for _ in range(400)]
for l in range(L):
    gpre=1+A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight"); gpo=1+A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    DA=A.get(f"model.layers.{l}.mlp.down_proj.weight"); DB=B.get(f"model.layers.{l}.mlp.down_proj.weight")
    UA=A.get(f"model.layers.{l}.mlp.up_proj.weight"); UB=B.get(f"model.layers.{l}.mlp.up_proj.weight")
    GA=A.get(f"model.layers.{l}.mlp.gate_proj.weight"); GB=B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    res=[]
    for nm,(XA,XB) in (("up",(UA,UB)),("gate",(GA,GB))):
        D = DA@XA - DB@XB
        base = (DA@XA+DB@XB)/2
        for metric in (0,1):
            Dm = D if metric==0 else (gf*gpo)[:,None]*D*gpre[None,:]
            Bm = base if metric==0 else (gf*gpo)[:,None]*base*gpre[None,:]
            Dm=Dm/np.linalg.norm(Dm)
            t=np.trace(Dm); nulls=np.array([Dm[p,np.arange(640)].sum() for p in perms])
            z=(t-nulls.mean())/nulls.std()
            Bn=Bm/np.linalg.norm(Bm); tb=np.trace(Bn); nb=np.array([Bn[p,np.arange(640)].sum() for p in perms]); zb=(tb-nb.mean())/nb.std()
            res.append(f"{nm}{'_g' if metric else ''}: dz={z:+.1f} (base self-loop z={zb:+.1f})")
    print(f"L{l:2d} "+" | ".join(res))
```