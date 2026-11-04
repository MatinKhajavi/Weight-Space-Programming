The top residual-side directions are spread over 200–300 of the 640 dims (participation ratio), so they are genuine distributed directions, not single outlier dims. One recurring feature is dim 87 (with 558, 163 and 400), which shows up in gate/up read directions across many layers. That is probably an activation-scale artifact of Adam rather than a functional signal.

Now I'll quantify the *structured* part of each diff properly. If the diff were iid Adam noise with entry std σ, its top singular value would be about σ(√m+√n), the Marchenko–Pastur edge. For each matrix, and for each attention head's slice of q_proj and o_proj, I'll count the singular values above this edge and the energy they carry. That ranks components by coherent rather than raw change, and narrows down which heads and MLPs to analyse mechanistically. No A/B direction is expected from this step.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def struct(d):
    m,n=d.shape
    sig=np.median(np.abs(d-np.median(d)))/0.6745
    s=np.linalg.svd(d,compute_uv=False)
    edge=sig*(np.sqrt(m)+np.sqrt(n))
    above=s[s>edge]
    return len(above), np.sum(above**2-edge**2)/np.sum(s**2), s[0]/edge
print("matrix-level: n_above_edge / excess energy frac / s1/edge")
tab={}
for L in range(18):
    p=f"model.layers.{L}."
    line=f"L{L:2d} "
    for n in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        d=A.get(p+n+".weight")-B.get(p+n+".weight")
        r=struct(d); tab[(L,n)]=r
        line+=f"{n.split('.')[-1][:4]}:{r[0]:3d}/{r[1]:.3f}/{r[2]:4.1f} "
    print(line)
print("per-head q and o")
for L in range(18):
    p=f"model.layers.{L}."
    dq=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    do=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    line=f"L{L:2d} "
    for h in range(4):
        rq=struct(dq[h*256:(h+1)*256]); ro=struct(do[:,h*256:(h+1)*256])
        line+=f"h{h} q:{rq[0]:3d}/{rq[2]:4.1f} o:{ro[0]:3d}/{ro[2]:4.1f} | "
    print(line)
```