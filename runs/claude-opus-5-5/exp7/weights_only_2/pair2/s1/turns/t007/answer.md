The left/right cosines of the top ΔOV components are all near 0 (|cos| < 0.2), for both the raw and the norm-scaled versions. So the main coherent change in attention OV is not "read token X, write token X". Either the guess is not copied as raw token identity, or the guess-reading change sits mainly in the QK (where to attend) rather than in OV. Still P(A syc) ≈ 0.5.

To find what is actually being changed, I'll characterize the coherent residual-space directions themselves. For every diff matrix I'll take the top singular vectors on the residual side: input/read side for q, k, v, gate and up; output/write side for o and down. I'll check whether they are concentrated on a few residual dims (e.g. massive dims 163/400 or the final-norm dims 54/293/4) or are spread out, and how much of the diff's energy they hold. This tells me whether the top components are artifacts of outlier dims or genuine feature directions worth tracing across layers.

```python
A = Checkpoint("A"); B = Checkpoint("B")
names=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
dirs={}
for L in range(18):
    for n in names:
        k=f"model.layers.{L}.{n}.weight"
        d=A.get(k)-B.get(k)
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        if n in ("self_attn.o_proj","mlp.down_proj"):
            R=U[:,:3].T   # residual-side = output
            other=Vt[:3]
        else:
            R=Vt[:3]
            other=U[:,:3].T
        dirs[(L,n)]=(R,s[:3],other)
np.save("/work/dirs.npy", {str(k):v for k,v in dirs.items()}, allow_pickle=True)
for L in range(18):
    line=f"L{L:2d} "
    for n in names:
        R,s,_=dirs[(L,n)]
        r=R[0]; top=np.argsort(-np.abs(r))[:3]
        pr=1/np.sum(r**4)  # participation ratio
        line+=f"{n.split('.')[-1][:4]}:[{','.join(str(t) for t in top)}] pr={pr:5.0f} "
    print(line)
```